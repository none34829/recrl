#!/usr/bin/env python
"""
Verify CTR lift from RLHF training using W&B API.

This script analyzes W&B runs to verify that CTR increases by at least 1.5 
percentage points from baseline to final epoch.

Usage:
  python verify_ctr_lift.py --project recrl_rlhf [--dataset books]
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def verify_ctr_lift(project, dataset=None, min_lift=0.015):
    """Verify CTR lift across RLHF runs."""
    try:
        import wandb
        api = wandb.Api()
        
        # Construct filter
        filters = {"tags": {"$in": [dataset]}} if dataset else {}
        
        # Get runs
        runs = api.runs(project, filters=filters)
        
        if not runs:
            print(f"No runs found for project '{project}'" + 
                  (f" with dataset '{dataset}'" if dataset else ""))
            return False
        
        results = []
        for run in runs:
            try:
                # Get history
                df = run.history(keys=["ctr"], pandas=True)
                if df.empty or "ctr" not in df.columns:
                    print(f"Run {run.name} has no CTR data")
                    continue
                
                # Compute lift
                initial_ctr = df["ctr"].iloc[0]
                final_ctr = df["ctr"].iloc[-1]
                lift = final_ctr - initial_ctr
                lift_pct = lift * 100  # Convert to percentage points
                
                # Store result
                results.append({
                    "run": run.name,
                    "dataset": run.config.get("dataset", "unknown"),
                    "initial_ctr": initial_ctr,
                    "final_ctr": final_ctr,
                    "lift": lift,
                    "lift_pct": lift_pct,
                    "passes": lift >= min_lift
                })
                
                # Print result
                status = "✓" if lift >= min_lift else "✗"
                print(f"{status} {run.name}: {lift_pct:.2f}pp lift ({initial_ctr:.4f} → {final_ctr:.4f})")
                
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")
        
        if not results:
            print("No valid runs found with CTR data")
            return False
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Summary statistics
        print(f"\n{'='*60}\nSUMMARY STATISTICS\n{'='*60}")
        print(f"Total runs analyzed: {len(df)}")
        print(f"Runs with sufficient lift: {df['passes'].sum()} / {len(df)}")
        print(f"Average lift: {df['lift_pct'].mean():.2f} percentage points")
        print(f"Median lift: {df['lift_pct'].median():.2f} percentage points")
        
        # Plot results if matplotlib is available
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(df["run"], df["lift_pct"], color=df["passes"].map({True: "green", False: "red"}))
            plt.axhline(y=min_lift*100, color="black", linestyle="--", label=f"Min threshold ({min_lift*100}pp)")
            plt.xlabel("Run")
            plt.ylabel("CTR Lift (percentage points)")
            plt.title("CTR Lift by Run")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.legend()
            
            # Save figure
            plt.savefig("ctr_lift_results.png")
            print("\nResults plot saved to ctr_lift_results.png")
        except Exception as e:
            print(f"\nCould not create plot: {e}")
        
        # Overall success condition
        success = df["passes"].all()
        
        if success:
            print(f"\n✓ All runs achieved minimum {min_lift*100:.1f} percentage point CTR lift!")
        else:
            print(f"\n✗ Some runs did not achieve minimum {min_lift*100:.1f} percentage point CTR lift.")
        
        return success
        
    except ImportError:
        print("Error: wandb package not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f"Error verifying CTR lift: {e}")
        return False

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Verify CTR lift from RLHF training")
    parser.add_argument("--project", default="recrl_rlhf", 
                       help="W&B project name (default: recrl_rlhf)")
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam"],
                       help="Filter by dataset (optional)")
    parser.add_argument("--min-lift", type=float, default=0.015,
                       help="Minimum CTR lift in absolute value (default: 0.015)")
    
    args = parser.parse_args()
    
    # Run verification
    success = verify_ctr_lift(
        project=args.project,
        dataset=args.dataset,
        min_lift=args.min_lift
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
