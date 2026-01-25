from typing import List, Dict, Any, Optional, Tuple, Union
import bisect
import numpy as np
import re
import math
import io
import struct
from itertools import product
from .utils import logger

def fit_integer_linear(
    y: np.ndarray,
    res_min_allowed: int,
    res_max_allowed: int
) -> Optional[Tuple[int, int]]:
    """
    整数线性拟合 y ≈ a*x + b，x=0..n-1。
    返回最佳整数 a, b，使最大残差绝对值最小，
    并且所有残差在 [res_min_allowed, res_max_allowed]。
    如果没有可行解返回 None。
    """
    n = len(y)
    x = np.arange(n, dtype=np.float64)

    # 浮点最小二乘解
    a = np.vstack([x, np.ones(n)]).T
    a_f, b_f = np.linalg.lstsq(a, y, rcond=None)[0]

    # 枚举四种整数组合
    a_candidates = [int(np.floor(a_f)), int(np.ceil(a_f))]
    b_candidates = [int(np.floor(b_f)), int(np.ceil(b_f))]

    best_a, best_b = None, None
    best_score = float('inf')

    for a, b in product(a_candidates, b_candidates):
        pred = a * x + b
        res = y - pred
        if res.min() < res_min_allowed or res.max() > res_max_allowed:
            continue
        score = max(abs(res.min()), abs(res.max()))
        if score < best_score:
            best_score = score
            best_a, best_b = a, b

    if best_a is None:
        return None
    return best_a, best_b


class SegmentedLinearPredictorCompressor:

    @classmethod
    def compress_names(cls, names: List[List[str]], name_pattern: str):
        """Compress names
        """

        if len(names[0]) == 0:
            return [], name_pattern

        name = names[0]
        same = True

        for i in range(1, len(names)):
            if names[i] != name:
                same = False
                break

        if same:
            it = iter(name)
            return [], re.sub(r"0", lambda _: str(next(it)), name_pattern)

        compressed_columns = []

        # zip(*names) 实现矩阵转置：
        # 从 m 个长度为 n 的列表，变成 n 个长度为 m 的元组
        # col_data 代表同一个数字位置在所有 event 中的值
        for col_data in zip(*names):
            # 转换为 numpy int32 数组
            arr = np.array(col_data, dtype=np.int64)

            # 检查是否全部一样
            # (arr == arr[0]).all() 是 numpy 中检查全等的快速方法
            if (arr == arr[0]).all():
                # 如果全一样，退化为单个值 (Python int)
                compressed_columns.append(int(arr[0]))
            else:
                # 否则保留 numpy array
                compressed_columns.append(arr)

        return compressed_columns, name_pattern

    @classmethod
    def decompress_names(cls, compressed_names, name_pattern: str, index: int) -> str:
        """Decompress names
        """

        if len(compressed_names) == 0:
            return name_pattern

        nums = compressed_names[index]
        it = iter(nums)
        return re.sub(r"0", lambda _: str(next(it)), name_pattern)

    @classmethod
    def compress_same_args(cls, args: Dict[str, List[Any]]) -> None:
        """Compress arguments that are the same for all examples.
        """

        for key, value in args.items():
            if not isinstance(value, list):
                continue
            if cls._all_same_args(value):
                args[key] = [value[0]]
            elif isinstance(value[0], int):
                args[key] = cls.compress_tss(value)

    @classmethod
    def decompress_same_args(cls, args: Dict[str, List[Any]], index: int) \
        -> Dict[str, Any]:
        """Decompress arguments that are the same for all examples.
        """

        if args is None:
            return None

        result = {}
        for key, value in args.items():
            if len(value) == 1:
                result[key] = value[0]
            else:
                result[key] = value[index]

        return result
    
    @classmethod
    def segment_linear_compress(cls, array: List[int]) -> Union[np.ndarray, Dict[str, Any]]:
        return np.asarray(array, dtype=np.int64)
    
    @classmethod
    def decompress_linear_segment(cls, compressed_array: Union[np.ndarray, Dict[str, Any]], index: int) -> int:
        if isinstance(compressed_array, np.ndarray):
            if compressed_array.size == 0:
                return None
            return int(compressed_array[index])
        else:
            logger.error(f"Not finished")

    @classmethod
    def compress_tss(cls, tss: List[int]) -> Dict[str, Any]:
        ts = np.asarray(tss, dtype=np.int64)
        return ts
        n = len(ts)

        baseline_bytes = n * 8
        precisions = [
            ("int8", -128, 127, 1),
            ("int16", -32768, 32767, 2),
            ("int32", -2147483648, 2147483647, 4)
        ]

        best_total_bytes = baseline_bytes
        best_segments = None

        for dtype, res_min, res_max, dtype_bytes in precisions:
            segments = []
            total_bytes = 0
            i = 0
            while i < n:
                # 尽量延长 segment
                last_fit = None
                for j in range(i+1, n+1):
                    seg = ts[i:j]
                    ab = fit_integer_linear(seg, res_min, res_max)
                    if ab is not None:
                        last_fit = (j, ab)
                    else:
                        break
                if last_fit is None:
                    # fallback 单点
                    j = i + 1
                    a, b = 0, int(ts[i])
                    residuals = np.array([0], dtype=np.int64)
                else:
                    j, (a, b) = last_fit
                    seg = ts[i:j]
                    pred = a * np.arange(len(seg), dtype=np.int64) + b
                    residuals = seg - pred
                    residuals = residuals.astype({1: np.int8, 2: np.int16, 4: np.int32}[dtype_bytes])

                metadata_bytes = 4+4+8+8  # start+length+a+b
                total_bytes += metadata_bytes + dtype_bytes * len(residuals)
                segments.append((i, j-i, a, b, residuals))
                i = j

            if total_bytes < best_total_bytes:
                # logger.info("succ, %d -> %d %s", baseline_bytes, total_bytes, dtype)
                best_total_bytes = total_bytes
                best_segments = segments

        if best_segments is None:
            return np.asarray(tss, dtype=np.int64)
        return best_segments

    @classmethod
    def compress_ids(cls, ids: List[str]):
        id_s, id_arr = cls._parse_id_list(ids)

        return id_s, id_arr

    @classmethod
    def _parse_id_list(cls, id_list: List[str]) -> Tuple[str, np.ndarray]:
        """
        Parse a list of ids into (prefix, numeric_array).

        Rules:
        - Either all ids are pure digits:       ["123", "456"]
        - Or all ids are prefix+digits:          ["f90", "f89"]
        - Mixed or invalid formats are rejected.

        Returns:
            prefix (str): common string prefix, "" if pure numeric
            nums   (np.ndarray[int32]): numeric parts
        """
        if not id_list:
            raise ValueError("id_list is empty")

        # Case 1: all pure digits
        if all(s.isdigit() for s in id_list):
            nums = np.array([int(s) for s in id_list], dtype=np.int32)
            return "", nums

        # Case 2: prefix + digits
        m = re.match(r"^([A-Za-z]+)(\d+)$", id_list[0])
        if not m:
            raise ValueError(f"Invalid id format: {id_list[0]}")

        prefix = m.group(1)

        nums = []
        for s in id_list:
            m = re.match(rf"^{prefix}(\d+)$", s)
            if not m:
                raise ValueError(f"Inconsistent id format: {s}")
            nums.append(int(m.group(1)))

        return prefix, np.array(nums, dtype=np.int32)


    @classmethod
    def _all_same_args(cls, args: List[Any]) -> bool:
        """Check if all arguments are the same
        """
        if len(args) == 0:
            return True

        for arg in args:
            if arg != args[0]:
                return False

        return True

    @classmethod
    def compress_ids(cls, ids: List[int]) -> List[Tuple[int, int, int]]:
        """Compress ids by identifying and encoding contiguous arithmetic progressions.
        
        Each arithmetic progression block is encoded as a 3-tuple:
        (initial_value, increment, starting_index)
        
        Example: [10, 12, 14, 15, 16] -> [(10, 2, 0), (15, 1, 3)]
        """

        if not ids:
            logger.info("IDs list is empty, returning empty list.")
            return []

        if not isinstance(ids[0], int):
            logger.info(f"IDs list is not a list of integers {ids}, returning original list.")
            return ids

        result = cls._find_arithmetic_progression_blocks(ids)

        logger.info(f"Compressed IDs: {result}")
        return result

    @classmethod
    def _all_same_names(cls, names: List[List[int]]) -> bool:
        """Check if all names are the same
        """
        if len(names) == 0:
            return True

        for name in names:
            if len(name) == 0:
                continue
            tmp = name[0]
            for n in name[1:]:
                if n!= tmp:
                    return False

        return True

    @classmethod
    def _compress_name(cls, name: List[int]):
        """Compress a name
        """
        result = cls._find_repeat_block(name)
        if len(result) * 2 >= len(name):
            result = name

        return result

    @classmethod
    def _find_repeat_block(cls, values: List[int]):
        """Find a block of repeated values
        """
        if len(values) < 2:
            return values

        result = []
        start = 0
        end = 1
        last = values[0]
        while end < len(values):
            if values[end] == last:
                end += 1
            else:
                result.append((last, start))
                start = end
                last = values[end]

        result.append((last, start))

        return result


    @classmethod
    def _find_arithmetic_progression_blocks(cls, values: List[int]) -> List[Tuple[int, int, int]]:
        """Find contiguous arithmetic progression blocks in a list of integers.
        
        Args:
            values: The list of integers (IDs).
            
        Returns:
            A list of 3-tuples: (initial_value, increment, starting_index).
        """
        if len(values) < 2:
            # For 0 or 1 element, treat it as a block with increment 0 (or just the value) 
            # and length 1. For simplicity in this scheme, we can define a single element 
            # as having an arbitrary increment (e.g., 0 or 1) as long as it's consistently 
            # interpreted during decompression. Using increment=1 is common.
            if len(values) == 1:
                # The interpretation of a single element block: 
                # (value, increment=0 or 1, start_index=0). 
                # Let's use increment 1 as a convention for single elements, 
                # or better, only compress sequences of length >= 2.
                # If we must compress a single element, (value, 0, 0) is safer 
                # for decompression: value + i*0.
                return [(values[0], 0, 0)]
            return []

        blocks = []

        # Current block tracking
        start_index = 0
        current_index = 1

        # The common difference (increment) is determined by the first two elements.
        # We need at least two elements to establish a potential AP.
        increment = values[1] - values[0]

        while current_index < len(values):
            # Calculate the difference for the current pair
            diff = values[current_index] - values[current_index - 1]

            if diff == increment:
                # The current element continues the arithmetic progression
                current_index += 1
            else:
                # The arithmetic progression is broken

                # Check if the block has length >= 2.
                # A length 1 block (start_index == current_index - 1) is just the first element.
                # If the progression breaks at index 'current_index',
                # the AP spans from 'start_index' to 'current_index - 1'.

                # Length of the current AP: current_index - start_index
                # We save the block only if it has length >= 2 OR if it's the end of the list.
                # A single element block is implicitly covered if the next element is different.

                # Always save the completed block (start_index to current_index - 1)
                initial_value = values[start_index]
                blocks.append((initial_value, increment, start_index))

                # Start a new block
                start_index = current_index - 1 
                # Now start_index points to the last element of the previous block, 
                # which is the start of the new block.

                if current_index + 1 < len(values):
                    # We have at least two more elements to determine the new increment
                    increment = values[current_index] - values[start_index]
                    current_index += 1 # Move to the third element of the new block (which is the current_index + 1)
                else:
                    # Only one element remains (values[current_index]), which is a single block
                    # Save the single remaining element block with increment 0 as convention
                    initial_value = values[start_index]
                    blocks.append((initial_value, 0, start_index))
                    start_index = current_index
                    increment = 0 # Dummy increment
                    current_index = len(values) # Exit loop
                    
        # After the loop, the last block needs to be saved.
        # This block spans from 'start_index' to 'len(values) - 1'.
        # We only save it if 'start_index' is still valid.
        if start_index < len(values):
            # The current_index reached len(values), the last AP is from start_index 
            # up to the end with the last calculated 'increment'.
            
            # Re-check the case where the list ends with a break.
            # Example: [10, 12, 14, 15]
            # 1. start=0, current=1. diff=2. current=2.
            # 2. current=2. diff(14-12)=2. current=3.
            # 3. current=3. diff(15-14)=1. diff != increment (2). BREAK.
            #    Save block (10, 2, 0) [10, 12, 14]. start=2. 
            #    current=3. start_index=2. element=14.
            #    We check current_index + 1 < len(values): 4 < 4 (False).
            #    ELSE block: Save (14, 0, 2). start=3. current=4. Exit.
            
            # Example: [10, 12, 14, 16]
            # 1. start=0, current=1. diff=2. current=2.
            # 2. current=2. diff(14-12)=2. current=3.
            # 3. current=3. diff(16-14)=2. current=4.
            # 4. current=4. Loop ends.
            #    Save remaining block (10, 2, 0). (NO! The loop should save the last block)
            
            # Let's adjust the logic slightly to be cleaner for the single element break case.
            
            # --- Re-implementing the loop logic to handle the single element more cleanly ---
            
            # Resetting blocks for clean implementation
            blocks = []
            
            # start_index: The index where the current AP block begins.
            start_index = 0
            
            while start_index < len(values):
                # An AP requires at least 2 elements to define an increment.
                if start_index + 1 >= len(values):
                    # Only one element remaining: treat as a block of length 1, increment 0
                    blocks.append((values[start_index], 0, start_index))
                    break

                # Determine the initial increment (common difference)
                initial_value = values[start_index]
                increment = values[start_index + 1] - initial_value
                
                # end_index is the index *after* the last element of the current AP.
                # Start checking from the third element (index start_index + 2).
                end_index = start_index + 2
                
                while end_index < len(values):
                    # Check if the next element continues the AP
                    current_diff = values[end_index] - values[end_index - 1]
                    
                    if current_diff == increment:
                        end_index += 1
                    else:
                        # AP broken. The block goes from start_index to end_index - 1
                        break
                
                # Save the completed AP block
                # The block spans from start_index up to end_index - 1
                blocks.append((initial_value, increment, start_index))
                
                # Start the next block at the index where the break occurred, or where the list ended
                start_index = end_index

            return blocks
