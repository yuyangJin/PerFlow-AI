"""Compressor implementations for trace compression.

This module defines abstract and concrete compressor classes used to
perform intra-rank and inter-rank compression on trace structures.
Each compressor follows a unified interface and provides both compression
and decompression methods.
"""

from __future__ import annotations
from typing import List, Dict, Union, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import math
import struct
from .trace import BaseTrace, Trace, CompressedTrace
from .node import BaseNode, Node, TemplateNode, RefNode
from .utils import logger

class Compressor(ABC):
    """Abstract base class for all compressors.

    Subclasses must implement methods for intra-rank and inter-rank
    compression and decompression.
    """

    @abstractmethod
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        """Perform intra-rank compression."""
        raise NotImplementedError

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        """Perform inter-rank compression."""
        raise NotImplementedError

    @abstractmethod
    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        """Perform intra-rank decompression."""
        raise NotImplementedError

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        """Perform inter-rank decompression."""
        raise NotImplementedError


class TemplateCompressor(Compressor):
    """Template-based compressor using subtree deduplication.

    This compressor identifies repeated subtrees in the execution trace
    (common in transformer layers and decoding loops) and replaces them
    with references to shared ``TemplateNode`` structures.
    """

    def __init__(self):
        super().__init__()
        self.templates: Dict[str, Union[TemplateNode, RefNode]] = {}
        self.next_template_id = 0

    def intra_compress(self, trace: BaseTrace, rank: str = "0") -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"

        self.templates = {}
        self.next_template_id = 0

        # rank -> pid -> tid -> node
        compressed_ranks: Dict[str, Dict[str, Dict[str, Union[Node, RefNode]]]] = {}

        logger.info("Intra compressing rank %s", rank)

        for r, pid, tid, node in trace.iter_nodes(rank):
            logger.info("Compressing pid %s tid %s", pid, tid)

            compressed_ranks.setdefault(str(r), {}).setdefault(str(pid), {})

            root = self._build_call_tree(node)
            root = self._find_template_node(root)
            compressed_ranks[str(r)][str(pid)][str(tid)] = root

        logger.info("There are %d templates", len(self.templates))

        return CompressedTrace(self.templates, compressed_ranks, trace.get_metadata())

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        raise NotImplementedError

    def _build_call_tree(self, node: Node) -> Node:
        events = sorted(node.events, key=lambda e: e.get_ts())

        roots: List[Node] = []
        stack: List[Node] = []

        for e in events:
            while stack:
                top_event = stack[-1].events[0]
                if top_event.get_ts() + top_event.get_dur() <= e.get_ts():
                    stack.pop()
                else:
                    break

            new_node = Node(events=[e])

            if stack:
                stack[-1].add_child(new_node)
            else:
                roots.append(new_node)

            stack.append(new_node)

        if len(roots) == 1:
            root = roots[0]
        else:
            root = Node()
            for r in roots:
                root.add_child(r)

        root = self._flatten_tree(root)

        return root

    def _group_same_children(self, node: BaseNode) -> Tuple[List[List[BaseNode]], List[BaseNode]]:
        children = node.get_children()
        n = len(children)
        used = [False] * n
        groups = []

        for i in range(n):
            if used[i]:
                continue

            current_group = [children[i]]

            for j in range(i+1, n):
                if used[j]:
                    continue

                if children[i].is_same_node(children[j]):
                    current_group.append(children[j])
                    used[j] = True

            if len(current_group) > 1:
                used[i] = True
                groups.append(current_group)

        unused = [children[i] for i in range(n) if not used[i]]

        return groups, unused

    def _find_node_in_templates(self, node: Node) -> Optional[TemplateNode]:
        for template in self.templates.values():
            if template.is_same_node(node):
                return template
        return None

    def _find_template_node(self, node: Node) -> Node:
        # there is no ref node

        if not hasattr(node, "id"): # only root template node has id
            tem = self._find_node_in_templates(node)
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes([node])

                return RefNode(tem, index)

        groups, _ = self._group_same_children(node)
        node_to_ref_node: Dict[Node, RefNode] = {}
        for group in groups:
            tem = self._find_node_in_templates(group[0])
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes(group)
                for n in group:
                    next_index = index + n.get_node_count()
                    node_to_ref_node[n] = RefNode(tem, index)
                    index = next_index
            else:
                tem = TemplateNode(group)
                tem.id = str(self.next_template_id)
                self.templates[str(self.next_template_id)] = tem
                self.next_template_id += 1
                index = 0
                for n in group:
                    next_index = index + n.get_node_count()
                    node_to_ref_node[n] = RefNode(tem, index)
                    index = next_index

                self._find_template_node(tem)

        new_children = []
        for child in node.get_children():
            if child in node_to_ref_node:
                new_children.append(node_to_ref_node[child])
            else:
                new_children.append(self._find_template_node(child))

        node.children = new_children

        return node

    def _compress_templates(self):
        raise NotImplementedError


    def _flatten_tree(self, node: Node):
        if (len(node.get_children())) == 0:
            return node

        while (len(node.get_children())) == 1:
            node.add_events(node.get_children()[0].get_events())
            new_children = node.get_children()[0].get_children()
            node.children = new_children

        new_children = []
        for child in node.get_children():
            new_children.append(self._flatten_tree(child))

        node.children = new_children

        return node

    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, Trace), "Compressed trace must be of type Trace"
        raise NotImplementedError

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, Trace), "Compressed trace must be of type Trace"
        raise NotImplementedError

class SegmentDeltaCompressor(Compressor):
    """Segmented delta compression.

    Applies segmented delta compression to timestamps using timestamp differences.
    Additionally, leverages the periodic patterns in the IDs embedded in names to
    achieve further compression.
    """

    def __init__(self):
        super().__init__()
        self.slope = 0.0  # 拟合直线的斜率
        self.intercept = 0.0  # 拟合直线的截距
        self.compressed_deltas = bytearray()  # 压缩后的差值数据
        self.event_count = 0  # 事件数量
        self.bits_per_delta = 0  # 每个差值使用的比特数


    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        pass

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        pass

    def _compress_timestamps(self, timestamps: List[int]) -> Dict:
        """
        TODO: 改为英文
        目前的实现是整段线性压缩，后面逐渐改为分段压缩时间戳，或尝试各种压缩方式
        """
        # # TODO:
        # # 1. 确定返回什么数据
        # # 2. 实现分段压缩算法
        # # 3. 实现分段解压算法，因为这里我不知道你如何设计数据结构，所以解压函数的声明没写
        # raise NotImplementedError
        
        if not timestamps:
            return {}
            
        self.event_count = len(timestamps)
        
        # 对时间戳进行线性拟合
        x = np.arange(self.event_count)
        y = np.array(timestamps, dtype=np.float64)
        
        # 使用最小二乘法进行线性拟合
        self.slope, self.intercept = np.polyfit(x, y, 1)
        
        # 计算每个时间戳与基线的差值
        deltas = []
        for i, ts in enumerate(timestamps):
            baseline = self.slope * i + self.intercept
            delta = ts - baseline
            deltas.append(delta)
        
        # 压缩差值数据
        self._compress_deltas_with_minimal_bits(deltas)
        
        # 返回压缩后的数据
        return self._pack_compressed_data()    

    def _compress_deltas_with_minimal_bits(self, deltas: List[float]):
        """
        使用最小比特位压缩差值数据
        
        参数:
            deltas: 差值数组
        """
        # 将差值转换为定点数表示（使用32位整数，16位小数）
        scaled_deltas = [int(d * 65536) for d in deltas]
        
        # 找到差值的最小值和最大值
        min_delta = min(scaled_deltas)
        max_delta = max(scaled_deltas)
        
        # 计算需要的比特数
        range_size = max_delta - min_delta
        self.bits_per_delta = max(1, math.ceil(math.log2(range_size + 1)))
        
        # 将差值偏移，使最小值为0
        offset_deltas = [d - min_delta for d in scaled_deltas]
        
        # 使用位打包压缩差值
        self._bit_pack_deltas(offset_deltas, min_delta)

    def _bit_pack_deltas(self, deltas: List[int], min_delta: int):
        """
        使用位打包压缩差值
        
        参数:
            deltas: 偏移后的差值数组
            min_delta: 原始差值的最小值
        """
        # 存储最小值（用于解压时恢复）
        self.compressed_deltas.extend(struct.pack('i', min_delta))
        
        # 存储每个差值使用的比特数
        self.compressed_deltas.append(self.bits_per_delta)
        
        # 计算每个字节可以存储的差值数量
        values_per_byte = 8 // self.bits_per_delta
        if values_per_byte == 0:
            values_per_byte = 1
        
        # 将差值打包到字节中
        current_byte = 0
        bits_used = 0
        
        for delta in deltas:
            # 确保差值不超过指定比特数能表示的范围
            masked_delta = delta & ((1 << self.bits_per_delta) - 1)
            
            # 将差值添加到当前字节
            current_byte = (current_byte << self.bits_per_delta) | masked_delta
            bits_used += self.bits_per_delta
            
            # 如果当前字节已满，将其添加到字节数组
            if bits_used >= 8:
                byte_to_store = current_byte >> (bits_used - 8)
                self.compressed_deltas.append(byte_to_store)
                current_byte &= (1 << (bits_used - 8)) - 1
                bits_used -= 8
        
        # 处理最后一个不完整的字节
        if bits_used > 0:
            # 左移使数据对齐到字节的高位
            current_byte <<= (8 - bits_used)
            self.compressed_deltas.append(current_byte)
    

    def _pack_compressed_data(self) -> Dict:
        """
        打包压缩数据
        
        返回:
            压缩后的数据字典
        """
        return {
            "event_count": self.event_count,
            "slope": self.slope,
            "intercept": self.intercept,
            "compressed_deltas": bytes(self.compressed_deltas),
            "bits_per_delta": self.bits_per_delta
        }
    
    # def decompress_all(self, compressed_data: Dict) -> List[int]:
    #     """
    #     解压缩所有时间戳
        
    #     参数:
    #         compressed_data: 压缩后的数据
            
    #     返回:
    #         解压缩后的时间戳数组
    #     """
    #     if not compressed_data:
    #         return []
        
    #     # 解析压缩数据
    #     self._unpack_compressed_data(compressed_data)
        
    #     # 解压缩所有时间戳
    #     timestamps = []
    #     for i in range(self.event_count):
    #         timestamps.append(self.get_event_timestamp(i))
        
    #     return timestamps
    
    # def _unpack_compressed_data(self, compressed_data: Dict):
    #     """
    #     解析压缩数据
        
    #     参数:
    #         compressed_data: 压缩后的数据
    #     """
    #     self.event_count = compressed_data["event_count"]
    #     self.slope = compressed_data["slope"]
    #     self.intercept = compressed_data["intercept"]
    #     self.compressed_deltas = compressed_data["compressed_deltas"]
    #     self.bits_per_delta = compressed_data["bits_per_delta"]

    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass

    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass
