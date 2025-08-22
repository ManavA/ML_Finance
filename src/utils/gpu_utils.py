
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import psutil
import GPUtil
import time
from contextlib import contextmanager

# Check for CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Check for Polars availability
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False


class GPUManager:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.gpu_info = self._get_gpu_info()
        
    def _get_optimal_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        # Multi-GPU: Select GPU with most free memory
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free = torch.cuda.mem_get_info()[0]
                free_memory.append(free)
            
            best_gpu = np.argmax(free_memory)
            return torch.device(f'cuda:{best_gpu}')
        
        return torch.device('cuda:0')
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {'available': False}
        
        info = {
            'available': True,
            'count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'name': torch.cuda.get_device_name(),
            'compute_capability': torch.cuda.get_device_capability(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved(),
        }
        
        # Add GPUtil info if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info['temperature'] = gpu.temperature
                info['utilization'] = gpu.load * 100
                info['memory_used'] = gpu.memoryUsed
                info['memory_free'] = gpu.memoryFree
        except:
            pass
        
        return info
    
    def optimize_batch_size(self, model_size_mb: float, 
                          sample_size_mb: float,
                          safety_factor: float = 0.8) -> int:
        if not torch.cuda.is_available():
            return 32
        
        # Get available memory
        free_memory = torch.cuda.mem_get_info()[0] / (1024**2)
        
        # Reserve memory for gradients and optimizer
        reserved = model_size_mb * 3
        
        # Available for batch
        available = (free_memory - reserved) * safety_factor
        
        # Calculate batch size
        batch_size = int(available / sample_size_mb)
        
        # Ensure power of 2 and reasonable bounds
        batch_size = min(max(batch_size, 8), 512)
        batch_size = 2 ** int(np.log2(batch_size))
        
        return batch_size
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_summary(self) -> str:
        if not torch.cuda.is_available():
            return "GPU not available"
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return (f"GPU Memory: {allocated:.1f}GB allocated, "
                f"{reserved:.1f}GB reserved, "
                f"{free:.1f}GB free, "
                f"{total:.1f}GB total")


class DataAccelerator:
    def __init__(self):
        self.use_cupy = CUPY_AVAILABLE
        self.use_polars = POLARS_AVAILABLE
        
        if self.use_cupy:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=8 * 1024**3)
    
    def pandas_to_cupy(self, df: pd.DataFrame) -> cp.ndarray:
        if not self.use_cupy:
            return df.values
        
        return cp.asarray(df.values)
    
    def cupy_to_pandas(self, arr: cp.ndarray, columns=None, index=None) -> pd.DataFrame:
        if not self.use_cupy:
            return pd.DataFrame(arr, columns=columns, index=index)
        
        return pd.DataFrame(cp.asnumpy(arr), columns=columns, index=index)
    
    def accelerate_rolling_window(self, data: np.ndarray, window: int, 
                                 func: str = 'mean') -> np.ndarray:
        if not self.use_cupy:
            # Fallback to NumPy
            if func == 'mean':
                return pd.Series(data).rolling(window).mean().values
            elif func == 'std':
                return pd.Series(data).rolling(window).std().values
            else:
                raise ValueError(f"Unknown function: {func}")
        
        # Use CuPy for GPU acceleration
        gpu_data = cp.asarray(data)
        result = cp.zeros_like(gpu_data)
        
        for i in range(window, len(data)):
            window_data = gpu_data[i-window:i]
            if func == 'mean':
                result[i] = cp.mean(window_data)
            elif func == 'std':
                result[i] = cp.std(window_data)
            elif func == 'sum':
                result[i] = cp.sum(window_data)
        
        return cp.asnumpy(result)
    
    def accelerate_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        if not self.use_cupy:
            return np.corrcoef(x, y)[0, 1]
        
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        
        corr = cp.corrcoef(x_gpu, y_gpu)[0, 1]
        return float(cp.asnumpy(corr))
    
    def use_polars_for_groupby(self, df: pd.DataFrame, 
                              group_col: str, 
                              agg_dict: Dict) -> pd.DataFrame:
        if not self.use_polars:
            return df.groupby(group_col).agg(agg_dict)
        
        # Convert to Polars
        pl_df = pl.from_pandas(df)
        
        # Perform groupby
        result = pl_df.group_by(group_col).agg([
            pl.col(col).agg_func() 
            for col, agg_func in agg_dict.items()
        ])
        
        # Convert back to pandas
        return result.to_pandas()


class GPUMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = []
        self.is_monitoring = False
        
    def start(self):
        self.start_time = time.time()
        self.is_monitoring = True
        self.metrics = []
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def record(self):
        if not self.is_monitoring:
            return
        
        metric = {
            'time': time.time() - self.start_time,
            'cpu_percent': psutil.cpu_percent(),
            'ram_used': psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            metric['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            metric['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metric['gpu_utilization'] = gpu.load * 100
                    metric['gpu_temperature'] = gpu.temperature
            except:
                pass
        
        self.metrics.append(metric)
    
    def stop(self):
        self.is_monitoring = False
    
    def report(self) -> Dict[str, Any]:
        if not self.metrics:
            return {}
        
        report = {
            'duration': time.time() - self.start_time,
            'samples': len(self.metrics),
        }
        
        # Calculate statistics
        for key in self.metrics[0].keys():
            if key != 'time':
                values = [m[key] for m in self.metrics if key in m]
                if values:
                    report[f'{key}_mean'] = np.mean(values)
                    report[f'{key}_max'] = np.max(values)
                    report[f'{key}_min'] = np.min(values)
        
        if torch.cuda.is_available():
            report['peak_gpu_memory'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        return report
    
    def print_report(self):
        report = self.report()
        
        print("\n" + "="*60)
        print("GPU MONITORING REPORT")
        print("="*60)
        
        print(f"Duration: {report.get('duration', 0):.1f} seconds")
        print(f"Samples: {report.get('samples', 0)}")
        
        if 'gpu_memory_allocated_mean' in report:
            print(f"\nGPU Memory:")
            print(f"  Mean: {report['gpu_memory_allocated_mean']:.2f} GB")
            print(f"  Peak: {report.get('peak_gpu_memory', 0):.2f} GB")
        
        if 'gpu_utilization_mean' in report:
            print(f"\nGPU Utilization:")
            print(f"  Mean: {report['gpu_utilization_mean']:.1f}%")
            print(f"  Max:  {report['gpu_utilization_max']:.1f}%")
        
        print(f"\nCPU Usage:")
        print(f"  Mean: {report.get('cpu_percent_mean', 0):.1f}%")
        print(f"  Max:  {report.get('cpu_percent_max', 0):.1f}%")
        
        print(f"\nRAM Usage:")
        print(f"  Mean: {report.get('ram_used_mean', 0):.1f}%")
        print(f"  Max:  {report.get('ram_used_max', 0):.1f}%")


@contextmanager
def gpu_memory_tracker():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"GPU Memory Used: {(end_memory - start_memory) / (1024**3):.2f} GB")
        print(f"Peak GPU Memory: {peak_memory / (1024**3):.2f} GB")


def optimize_model_for_gpu(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.cuda.is_available():
        return model
    
    # Move to GPU
    model = model.cuda()
    
    # Enable cudnn optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Compile model if PyTorch 2.0+
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return model


def get_optimal_num_workers() -> int:
    cpu_count = psutil.cpu_count(logical=False)
    
    if torch.cuda.is_available():
        return min(cpu_count, 8)
    else:
        return max(1, cpu_count - 2)


# Utility functions for quick checks
def check_gpu_availability():
    print("="*60)
    print("GPU AVAILABILITY CHECK")
    print("="*60)
    
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"CuPy: {CUPY_AVAILABLE}")
    print(f"Polars: {POLARS_AVAILABLE}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"    Compute: {props.major}.{props.minor}")
    
    print("\nRecommendations:")
    if not torch.cuda.is_available():
        print("  ⚠️ No GPU detected - training will be slow")
    if not CUPY_AVAILABLE:
        print("  ⚠️ CuPy not installed - run: pip install cupy-cuda11x")
    if not POLARS_AVAILABLE:
        print("  ⚠️ Polars not installed - run: pip install polars")
    
    if torch.cuda.is_available() and CUPY_AVAILABLE and POLARS_AVAILABLE:
        print("  ✅ All acceleration libraries available!")


if __name__ == '__main__':
    # Run availability check
    check_gpu_availability()
    
    # Test GPU manager
    if torch.cuda.is_available():
        manager = GPUManager()
        print(f"\n{manager.get_memory_summary()}")
        
        # Test optimal batch size calculation
        batch_size = manager.optimize_batch_size(
            model_size_mb=500,
            sample_size_mb=1
        )
        print(f"Recommended batch size: {batch_size}")