#!/usr/bin/env python3
"""
Pilot Data Collection Script
Runs a smaller version of the massive data collection to validate the system.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from collect_massive_research_data import MassiveDataCollector


def run_pilot_collection():
    """Run a pilot data collection with reduced parameters."""
    print("🚁 PILOT DATA COLLECTION FOR OPTION A")
    print("Testing the massive data collection system")
    print("=" * 50)
    
    # Pilot configuration - much smaller for validation
    pilot_collector = MassiveDataCollector(
        target_runs=5,  # 5 runs per config (instead of 30)
        dimensions=[2, 5],  # Only 2 dimensions (instead of 5)
        max_evaluations=200,  # Smaller budget for speed
        output_dir="pilot_research_data",
        parallel_workers=2  # Conservative
    )
    
    print(f"🎯 Pilot Configuration:")
    print(f"  Total problems: {len(pilot_collector.problem_instances)}")
    print(f"  Algorithms: {len(pilot_collector.algorithms)}")
    print(f"  Runs per config: {pilot_collector.target_runs}")
    print(f"  Expected total runs: {len(pilot_collector.problem_instances) * len(pilot_collector.algorithms) * pilot_collector.target_runs:,}")
    
    # Run pilot collection with smaller batch size
    print(f"\n🚀 Starting pilot collection...")
    dataset_path = pilot_collector.run_massive_collection(batch_size=5)
    
    print(f"\n✅ Pilot collection completed!")
    print(f"📊 Dataset saved: {dataset_path}")
    
    # Basic validation
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        print(f"\n📈 Pilot Results Summary:")
        print(f"  Total runs: {len(df):,}")
        print(f"  Unique problems: {df.groupby(['problem_name', 'problem_dimension']).ngroups}")
        print(f"  Algorithms tested: {list(df['algorithm_name'].unique())}")
        print(f"  Success rate: {(~df['error'].notna()).mean():.1%}" if 'error' in df.columns else "  All runs successful")
        
        # Show performance summary
        if 'best_fitness' in df.columns:
            performance_summary = df.groupby('algorithm_name')['best_fitness'].agg(['mean', 'std', 'min'])
            print(f"\n🏆 Algorithm Performance Summary:")
            print(performance_summary)
        
        print(f"\n🎯 System validated! Ready for full research collection.")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Main pilot function."""
    success = run_pilot_collection()
    
    if success:
        print("\n🚀 READY FOR FULL RESEARCH COLLECTION!")
        print("Next steps:")
        print("  1. Run: python collect_massive_research_data.py")
        print("  2. Expected ~6,000+ runs for full research dataset")
        print("  3. Train transformer models on expanded dataset")
        print("  4. Prepare research publication")
    else:
        print("\n❌ Pilot failed. Please check and fix issues.")


if __name__ == "__main__":
    main() 