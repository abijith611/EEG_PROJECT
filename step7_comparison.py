import numpy as np
import step5_decoding  # Imports our corrected function

def run_comparative_analysis(epochs_binned, full_df, player_num):
    print(f"--- [step7_comparison] Analyzing Player {player_num} ---\n")
    
    # 1. Define Columns
    p_own = f'player{player_num}_resp'
    p_opp = f'player{2 if player_num==1 else 1}_resp'
    
    # 2. Extract Labels
    y_own_curr = full_df[p_own].values
    y_opp_curr = full_df[p_opp].values
    y_own_prev = full_df[p_own].shift(1).values
    y_opp_prev = full_df[p_opp].shift(1).values
    
    # 3. Define Tasks
    tasks = {
        'Own Current': y_own_curr,
        'Opponent Current': y_opp_curr,
        'Own Previous': y_own_prev,
        'Opponent Previous': y_opp_prev
    }
    
    # 4. Define Outcome Mask (Winner vs Loser)
    outcome_col = f'player{player_num}_outcome'
    outcomes = full_df[outcome_col].values
    
    conditions = {
        'Winner': (outcomes == 'win'),
        'Loser': (outcomes == 'loss') 
    }
    
    results = {'Winner': {}, 'Loser': {}}
    
    # 5. Nested Loop: Condition (Win/Loss) -> Task (Own/Opp)
    for cond_name, mask in conditions.items():
        print(f"  Condition: {cond_name} (Trials: {np.sum(mask)})")
        
        # Filter Data for this condition (e.g., only Winning trials)
        X_cond = epochs_binned[mask]
        
        for task_name, y_full in tasks.items():
            # Filter Labels
            y_cond = y_full[mask]
            
            # Clean NaNs (caused by .shift(1))
            valid_idx = ~np.isnan(y_cond)
            X_final = X_cond[valid_idx]
            y_final = y_cond[valid_idx]
            
            if len(y_final) < 10:
                # print(f"    Skipping {task_name}: Not enough data.")
                results[cond_name][task_name] = []
                continue
                
            # DECODE using the NEW Step 5 function
            # Use 5 folds here because subsetting by Win/Loss reduces data size
            _, mean_scores, _, _ = step5_decoding.run_svm_decoding_with_pseudotrials(
                X_final, y_final, 
                n_folds=5, n_average=4, n_repeats=20
            )
            
            results[cond_name][task_name] = mean_scores

    return results