import json
from pathlib import Path
from typing import Dict, List, Optional, Any

class DeviceController:
    """Manager for device specifications with interactive device selection."""
    
    def __init__(self, config_path: str = "device_specs.json"):
        # 获取当前文件所在目录，而不是运行目录
        base_dir = Path(__file__).resolve().parent
        self.config_path = base_dir / config_path
        self.device_specs = self._load_device_specs()
    
    def _load_device_specs(self) -> Dict:
        """Load device specifications from JSON file or use defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {self.config_path}, using default specs")
        
        # Default device specifications
        return {
            "GPU": {
                "NVIDIA_H100": {
                    "SXM5_80GB": {
                        "compute": {
                            "FP64_TFLOPS": 60, "FP32_TFLOPS": 1000, 
                            "FP16_TFLOPS": 2000, "FP8_TFLOPS": 4000, "INT8_TOPS": 4000
                        },
                        "memory_bandwidth_TBps": 3
                    },
                    "PCIe_80GB": {
                        "compute": {
                            "FP64_TFLOPS": 48, "FP32_TFLOPS": 800,
                            "FP16_TFLOPS": 3200, "FP8_TFLOPS": 3200, "INT8_TOPS": 3200
                        },
                        "memory_bandwidth_TBps": 2
                    }
                },
                "NVIDIA_A100": {
                    "SXM4_80GB": {
                        "compute": {
                            "FP64_TFLOPS": 19.5, "FP32_TFLOPS": 312,
                            "FP16_TFLOPS": 624, "INT8_TOPS": 1248
                        },
                        "memory_bandwidth_TBps": 2.0
                    },
                    "PCIe_80GB": {
                        "compute": {
                            "FP64_TFLOPS": 19.5, "FP32_TFLOPS": 312,
                            "FP16_TFLOPS": 624, "INT8_TOPS": 1248
                        },
                        "memory_bandwidth_TBps": 2.0
                    }
                }
            },
            "CPU": {
                "Intel_Xeon_Platinum_8480": {
                    "compute": {"FP64_TFLOPS": 2.5, "FP32_TFLOPS": 5.0, "FP16_TFLOPS": 10.0},
                    "memory_bandwidth_TBps": 0.3
                },
                "AMD_EPYC_9754": {
                    "compute": {"FP64_TFLOPS": 3.0, "FP32_TFLOPS": 6.0, "FP16_TFLOPS": 12.0},
                    "memory_bandwidth_TBps": 0.35
                }
            }
        }
    
    def save_device_specs(self):
        """Save device specifications to JSON file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.device_specs, f, indent=2)
        print(f"Device specifications saved to {self.config_path}")
    
    def list_device_categories(self) -> List[str]:
        """List available device categories."""
        return list(self.device_specs.keys())
    
    def list_devices(self, category: str) -> List[str]:
        """List available devices in a category."""
        return list(self.device_specs.get(category, {}).keys())
    
    def list_variants(self, category: str, device: str) -> List[str]:
        """List available variants for a device."""
        return list(self.device_specs.get(category, {}).get(device, {}).keys())
    
    def get_device_spec(self, category: str, device: str, variant: str) -> Dict:
        """Get specifications for a specific device variant."""
        return self.device_specs.get(category, {}).get(device, {}).get(variant, {})
    
    def interactive_device_selection(self):
        """Interactive device selection process."""
        print("=" * 60)
        print("Device Selection")
        print("=" * 60)
        
        # Step 1: Choose device category
        categories = self.list_device_categories()
        print("\nAvailable device categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        while True:
            try:
                choice = input(f"\nSelect device category (1-{len(categories)}), or 'a' to add new: ").strip()
                if choice.lower() == 'a':
                    self._add_new_device_category()
                    categories = self.list_device_categories()
                    continue
                
                category_idx = int(choice) - 1
                if 0 <= category_idx < len(categories):
                    selected_category = categories[category_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Step 2: Choose device
        devices = self.list_devices(selected_category)
        print(f"\nAvailable {selected_category} devices:")
        for i, device in enumerate(devices, 1):
            print(f"{i}. {device}")
        
        while True:
            try:
                choice = input(f"\nSelect device (1-{len(devices)}), 'b' to go back, or 'a' to add new: ").strip()
                if choice.lower() == 'b':
                    return self.interactive_device_selection()
                elif choice.lower() == 'a':
                    self._add_new_device(selected_category)
                    devices = self.list_devices(selected_category)
                    continue
                
                device_idx = int(choice) - 1
                if 0 <= device_idx < len(devices):
                    selected_device = devices[device_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Step 3: Choose variant
        variants = self.list_variants(selected_category, selected_device)
        print(f"\nAvailable variants for {selected_device}:")
        for i, variant in enumerate(variants, 1):
            print(f"{i}. {variant}")
        
        while True:
            try:
                choice = input(f"\nSelect variant (1-{len(variants)}), 'b' to go back, or 'a' to add new: ").strip()
                if choice.lower() == 'b':
                    return self.interactive_device_selection()
                elif choice.lower() == 'a':
                    self._add_new_variant(selected_category, selected_device)
                    variants = self.list_variants(selected_category, selected_device)
                    continue
                
                variant_idx = int(choice) - 1
                if 0 <= variant_idx < len(variants):
                    selected_variant = variants[variant_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get the selected device specifications
        device_spec = self.get_device_spec(selected_category, selected_device, selected_variant)
        
        print(f"\nSelected: {selected_category} {selected_device} {selected_variant}")
        print("Specifications:")
        print(json.dumps(device_spec, indent=2))
        
        return {
            'type': selected_category.lower(),
            'model': f"{selected_device} {selected_variant}",
            'specs': device_spec
        }
    
    def _add_new_device_category(self):
        """Guide user to add a new device category."""
        print("\nAdding new device category...")
        category = input("Enter new device category name (e.g., GPU, CPU, NPU): ").strip()
        if category and category not in self.device_specs:
            self.device_specs[category] = {}
            print(f"Added new category: {category}")
            self.save_device_specs()
        else:
            print("Category already exists or invalid name.")
    
    def _add_new_device(self, category: str):
        """Guide user to add a new device to a category."""
        print(f"\nAdding new device to {category}...")
        device = input("Enter new device name: ").strip()
        if device and device not in self.device_specs[category]:
            self.device_specs[category][device] = {}
            print(f"Added new device: {device}")
            self.save_device_specs()
            
            # Option to add a variant immediately
            if input("Add a variant now? (y/n): ").lower() == 'y':
                self._add_new_variant(category, device)
        else:
            print("Device already exists or invalid name.")
    
    def _add_new_variant(self, category: str, device: str):
        """Guide user to add a new variant to a device."""
        print(f"\nAdding new variant to {device}...")
        variant = input("Enter variant name: ").strip()
        if variant and variant not in self.device_specs[category][device]:
            print("\nPlease enter the device specifications:")
            specs = {}
            
            # Compute specifications
            compute = {}
            print("Compute performance (TFLOPS/TOPS):")
            for precision in ['FP64', 'FP32', 'FP16', 'FP8', 'INT8']:
                try:
                    value = float(input(f"{precision}: ").strip() or 0)
                    if value > 0:
                        compute[f"{precision}_TFLOPS"] = value
                except ValueError:
                    print(f"Invalid value for {precision}, skipping")
            
            specs['compute'] = compute
            
            # Memory bandwidth
            try:
                mem_bw = float(input("Memory bandwidth (TB/s): ").strip() or 0)
                if mem_bw > 0:
                    specs['memory_bandwidth_TBps'] = mem_bw
            except ValueError:
                print("Invalid memory bandwidth value")
            
            self.device_specs[category][device][variant] = specs
            print(f"Added variant: {variant}")
            self.save_device_specs()
        else:
            print("Variant already exists or invalid name.")