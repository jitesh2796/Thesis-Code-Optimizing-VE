import pandas as pd
import json

def evaluate_model_performance(jsonl_path, right_answers_list):
    # Step 1: Load main jsonl file
    with open(jsonl_path, 'r') as f:
        main_data = [json.loads(line) for line in f]
    df_main = pd.DataFrame(main_data)

    # Step 2: Load right_answers_list (list of dicts with 'id' and 'right_answer')
    df_right = pd.DataFrame(right_answers_list)
    df_right.rename(columns={"answer":"right_answer"}, inplace =True)

    # Step 3: Merge on 'id'
    df = df_main.merge(df_right[['id', 'right_answer']], on='id', how='left')

    # Step 4: Create match_flag
    def get_match(row):
        if row['high_consistency']:
            return int(row['answer'].strip() == row['right_answer'].strip())
        else:
            return int(row.get('new_answer', '').strip() == row['right_answer'].strip())
    
    df['match_flag'] = df.apply(get_match, axis=1)

    # Step 5: Metrics
    exact_match_pct = 100 * df['match_flag'].mean()
    verify_pct = 100 * (~df['high_consistency']).mean()
    avg_time = df['time_taken_sec'].mean()

    return df, {
        'exact_match_percentage': round(exact_match_pct, 2),
        'verify_percentage': round(verify_pct, 2),
        'average_time_per_record_sec': round(avg_time, 2)
    }


def evaluate_model_performance_v2(jsonl_path):
    # Step 1: Read jsonl into DataFrame
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    # Step 2: Create match_flag
    def get_match(row):
        if row['high_consistency']:
            return int(row['answer'].strip().lower() == row['right_answer'].strip().lower())
        else:
            return int(row.get('new_answer', '').strip().lower() == row['right_answer'].strip().lower())
    
    df['match_flag'] = df.apply(get_match, axis=1)

    # Step 3: Exact match %
    exact_match_pct = 100 * df['match_flag'].mean()

    # Step 4: Verify % (high_consistency == False)
    verify_pct = 100 * (~df['high_consistency']).mean()

    # Step 5: Avg time taken
    avg_time = df['time_taken_sec'].mean()

    return df, {
        'exact_match_percentage': round(exact_match_pct, 2),
        'verify_percentage': round(verify_pct, 2),
        'average_time_per_record_sec': round(avg_time, 2)
    }