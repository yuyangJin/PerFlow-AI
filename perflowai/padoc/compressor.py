from __future__ import annotations
from .trace import BaseTrace, Trace, CompressedTrace
from .node import BaseNode, Node, TemplateNode, RefNode
from .event import Event, MergeEvent
from .utils import logger
from typing import List, Dict, Union, Optional, Tuple
from abc import ABC, abstractmethod


class Compressor(ABC):
    
    @abstractmethod
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError
    
    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError

    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError
    
    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError


class TemplateCompressor(Compressor):

    def intra_compress(self, trace: BaseTrace, rank: int = 0) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"

        self.templates: Dict[int, BaseNode] = {}
        self.next_template_id = 0

        # rank -> pid -> tid -> node
        compressed_ranks: Dict[int, Dict[int, Dict[int, BaseNode]]] = {}
        
        # TODO: compress when building call tree
        # 1. build call tree
        # 1.1 get each pid and tid node
        # 1.2 build call tree for each pid and tid node
        for r, pid, tid, node in trace.iter_nodes(rank):

            compressed_ranks.setdefault(r, {}).setdefault(pid, {})

            root = self.build_call_tree(node)
            root = self._find_template_node(root)
            compressed_ranks[r][pid][tid] = root

            # groups, unused = self.group_same_children(root)
            # for i, group in enumerate(groups):
            #     name = group[0].get_events()[0].get_name()
            #     safe_name = name.split("/")[-1].split(":")[0]
            #     import re
            #     safe_name = re.sub(r"[^0-9a-zA-Z_-]", "_", safe_name)
            #     file_path = f"{safe_name}_group_{i}.json"
            #     import json
            #     with open(file=file_path, mode="w") as f:
            #         json.dump(group[0].to_dict(), f, indent=2)
            #     logger.info(f"compressing {len(group)} nodes, name: {group[0].get_events()[0].get_name()}")

            # for i, n in enumerate(unused):
            #     logger.info(f"keeping {n.get_events()[0].get_name()}")
            #     name = n.get_events()[0].get_name()
            #     safe_name = name.split("/")[-1].split(":")[0]
            #     import re
            #     safe_name = re.sub(r"[^0-9a-zA-Z_-]", "_", safe_name)
            #     file_path = f"{safe_name}_unused_{i}.json"
            #     import json
            #     with open(file=file_path, mode="w") as f:
            #         json.dump(n.to_dict(), f, indent=2)

        # TODO:
        # 2. compress call tree
        # 2.1 combine nodes if there are multiple same nodes
        # 2.2 find template nodes and compress them

        logger.info(f"compressed {len(compressed_ranks)} ranks")
        logger.info(f"compressed {len(self.templates)} templates")

        return CompressedTrace(self.templates, compressed_ranks, trace.get_metadata())

    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        assert isinstance(trace, Trace), "Trace must be of type Trace"
        # TODO
        pass

    def build_call_tree(self, node: Node) -> Node:
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

        logger.info(f"build call tree: {len(roots)} roots")
        if len(roots) == 1:
            root = roots[0]
        else:
            root = Node()
            for r in roots:
                root.add_child(r)

        root = self._flatten_tree(root)

        return root
    
    def group_same_children(self, node: BaseNode) -> Tuple[List[List[BaseNode]], List[BaseNode]]:
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
                    logger.debug(f"found same node: {children[i].get_events()[0].get_name()} and {children[j].get_events()[0].get_name()}")
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
        
        if isinstance(node, TemplateNode):
            logger.error(f"node is already a template node: {node.get_events()[0].get_name()}")
            raise ValueError("node is already a template node")
        
        tem = self._find_node_in_templates(node)
        if tem is not None:
            index = tem.get_node_count()
            tem.add_nodes([node])

            return RefNode(tem, index)
        
        groups, unused = self.group_same_children(node)
        node_to_ref_node: Dict[Node, RefNode] = {}
        for group in groups:
            tem = self._find_node_in_templates(group[0])
            if tem is not None:
                index = tem.get_node_count()
                tem.add_nodes(group)
                for n in group:
                    node_to_ref_node[n] = RefNode(tem, index)
                    index += 1
            else:
                tem = TemplateNode(group)
                self.templates[self.next_template_id] = tem
                self.next_template_id += 1
                index = 0
                for n in group:
                    node_to_ref_node[n] = RefNode(tem, index)
                    index += 1

        new_children = []
        for child in node.get_children():
            if child in node_to_ref_node:
                new_children.append(node_to_ref_node[child])
            else:
                new_children.append(self._find_template_node(child))

        node.children = new_children

        return node
        

    def _flatten_tree(self, node: Node):
        if (len(node.get_children())) == 0:
            return node
        
        while (len(node.get_children())) == 1:
            logger.debug(f"flattening {node.get_children()[0].get_events()[0].get_name()} to {node.get_events()[-1].get_name()}")
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
        # TODO
        pass
    
    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        assert isinstance(compressed_trace, Trace), "Compressed trace must be of type Trace"
        # TODO
        pass

class SegmentDeltaCompressor(Compressor):
    def intra_compress(self, trace: BaseTrace) -> BaseTrace:
        pass
    
    def inter_compress(self, trace: BaseTrace) -> BaseTrace:
        pass

    def _compress_timestamps(self, timestamps: List[int]):
        """
        TODO: 改为英文
        分段压缩时间戳
        """
        # TODO: 
        # 1. 确定返回什么数据
        # 2. 实现分段压缩算法
        # 3. 实现分段解压算法，因为这里我不知道你如何设计数据结构，所以解压函数的声明没写
        pass


    
    def intra_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass
    
    def inter_decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        pass