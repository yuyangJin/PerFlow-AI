from typing import List, Dict, Any, Optional, Tuple, Union
import bisect
import numpy as np
import re
from .utils import logger

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

        # print(names, name_pattern)

        return names, name_pattern

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
            if cls._all_same_args(value):
                args[key] = [value[0]]

    @classmethod
    def decompress_same_args(cls, args: Dict[str, List[Any]], index: int) \
        -> Dict[str, Any]:
        """Decompress arguments that are the same for all examples.
        """

        result = {}
        for key, value in args.items():
            if len(value) == 1:
                result[key] = value[0]
            else:
                result[key] = value[index]

        return result

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
