"""
Benchmark script for MediaPipe gaze tracking.
Run with: python benchmark_gaze_trackers.py
"""

from collections import deque
import cv2
import time
import numpy as np
from pathlib import Path
import json
import argparse

try:
    from gaze_tracking.gaze_tracking_mediapipe import GazeTrackingMediaPipe
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("WARNING: MediaPipe not available. Install with: pip install mediapipe")


class BenchmarkGazeTracking:
    """Benchmark gaze tracking implementations."""
    
    def __init__(self, duration: float = 30.0, output_path: str = 'benchmark_results.json'):
        self.duration = duration
        self.output_path = output_path
        self.results = {}
        self._h_hist = deque(maxlen=3)
        self._v_hist = deque(maxlen=3)

        self._left_corner_l = None   # left eye: left corner in IMAGE coords
        self._left_corner_r = None   # left eye: right corner in IMAGE coords
        self._right_corner_l = None  # right eye: left corner in IMAGE coords
        self._right_corner_r = None  # right eye: right corner in IMAGE coords

        self._left_top = None
        self._left_bottom = None
        self._right_top = None
        self._right_bottom = None

    def benchmark_implementation(self, impl_name: str, gaze_tracker):
        """Benchmark a single gaze tracker implementation."""
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {impl_name}")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"ERROR: Camera not available for {impl_name}")
            return None
        
        # Warm up
        print("Warming up...")
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                gaze_tracker.refresh(frame)
        
        # Run benchmark
        print(f"Running benchmark for {self.duration} seconds...")
        start_time = time.time()
        frame_count = 0
        pupil_detected_count = 0
        head_pose_count = 0
        latencies = []
        
        while time.time() - start_time < self.duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Measure latency
            t0 = time.time()
            gaze_tracker.refresh(frame)
            latency = (time.time() - t0) * 1000  # ms
            latencies.append(latency)
            
            if gaze_tracker.pupils_located:
                pupil_detected_count += 1
            
            if gaze_tracker.head_pose:
                head_pose_count += 1
            
            # Display progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                det_rate = pupil_detected_count / frame_count if frame_count > 0 else 0
                print(f"  Frames: {frame_count}, FPS: {fps:.1f}, Pupil detect: {det_rate:.1%}")
        
        cap.release()
        
        # Compute stats
        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        pupil_rate = pupil_detected_count / frame_count if frame_count > 0 else 0
        head_pose_rate = head_pose_count / frame_count if frame_count > 0 else 0
        
        latencies = np.array(latencies)
        
        stats = {
            'implementation': impl_name,
            'duration_s': total_time,
            'frames_processed': frame_count,
            'fps': fps,
            'pupil_detection_rate': pupil_rate,
            'head_pose_estimation_rate': head_pose_rate,
            'latency_ms': {
                'mean': float(np.mean(latencies)) if len(latencies) > 0 else 0,
                'median': float(np.median(latencies)) if len(latencies) > 0 else 0,
                'std': float(np.std(latencies)) if len(latencies) > 0 else 0,
                'min': float(np.min(latencies)) if len(latencies) > 0 else 0,
                'max': float(np.max(latencies)) if len(latencies) > 0 else 0,
                'p95': float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0,
                'p99': float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0,
            }
        }
        
        self.results[impl_name] = stats
        
        # Print results
        print(f"\nResults for {impl_name}:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Frames processed: {frame_count}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Pupil detection rate: {pupil_rate:.1%}")
        print(f"  Head pose estimation rate: {head_pose_rate:.1%}")
        print(f"  Mean latency: {stats['latency_ms']['mean']:.2f} ms")
        print(f"  Median latency: {stats['latency_ms']['median']:.2f} ms")
        print(f"  Latency std: {stats['latency_ms']['std']:.2f} ms")
        print(f"  Latency p95: {stats['latency_ms']['p95']:.2f} ms")
        print(f"  Latency p99: {stats['latency_ms']['p99']:.2f} ms")
        
        return stats
    
    def compare_results(self):
        """Compare results across implementations."""
        if len(self.results) < 2:
            print("\nNeed at least 2 implementations to compare")
            return
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        impls = list(self.results.keys())
        
        # FPS comparison
        print("\nFPS (higher is better):")
        for impl in impls:
            fps = self.results[impl]['fps']
            print(f"  {impl}: {fps:.1f} fps")
        
        if len(impls) == 2:
            if self.results[impls[0]]['fps'] > 0:
                speedup = self.results[impls[1]]['fps'] / self.results[impls[0]]['fps']
                print(f"  → {impls[1]} is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        # Latency comparison
        print("\nMean latency (lower is better):")
        for impl in impls:
            lat = self.results[impl]['latency_ms']['mean']
            print(f"  {impl}: {lat:.2f} ms")
        
        if len(impls) == 2:
            improvement = (self.results[impls[0]]['latency_ms']['mean'] - 
                          self.results[impls[1]]['latency_ms']['mean'])
            if self.results[impls[0]]['latency_ms']['mean'] > 0:
                pct = (improvement / self.results[impls[0]]['latency_ms']['mean']) * 100
                if improvement > 0:
                    print(f"  → {impls[1]} is {pct:.1f}% faster")
                else:
                    print(f"  → {impls[0]} is {-pct:.1f}% faster")
        
        # Detection rates
        print("\nPupil detection rate (higher is better):")
        for impl in impls:
            rate = self.results[impl]['pupil_detection_rate']
            print(f"  {impl}: {rate:.1%}")
    
    def save_results(self):
        """Save results to JSON."""
        output_path = Path(self.output_path)
        output_path.write_text(json.dumps(self.results, indent=2))
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark gaze tracking implementations')
    parser.add_argument('--duration', type=float, default=30.0, help='Benchmark duration in seconds')
    parser.add_argument('--output', default='benchmark_results.json', help='Output JSON file')
    args = parser.parse_args()
    
    benchmark = BenchmarkGazeTracking(duration=args.duration, output_path=args.output)
    
    # Run MediaPipe benchmark
    if HAS_MEDIAPIPE:
        gaze_mediapipe = GazeTrackingMediaPipe()
        benchmark.benchmark_implementation('MediaPipe', gaze_mediapipe)
    else:
        print("MediaPipe not installed; skipping benchmark")
    
    # Compare
    benchmark.compare_results()
    benchmark.save_results()


if __name__ == '__main__':
    main()