import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

FILE = 'l1_day.csv'
df = pd.read_csv(FILE, parse_dates=['ts_event'])

df = df.sort_values(by='ts_event')
df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')

grouped = df.groupby('ts_event')

ORDER_SIZE = 5000

l_over = np.array([0.1, 0.5, 1.0])
l_under = np.array([0.1, 0.5, 1.0])
theta = np.array([0.0, 0.1, 0.2])

param_df = pd.DataFrame(np.array(np.meshgrid(l_over, l_under, theta)).T.reshape(-1, 3),
                         columns=['lambda_over', 'lambda_under', 'theta_queue'])
PARAM_GRID = param_df.values.tolist()

def allocate(order_size, venues, lam_o, lam_u, theta):
    step = 100
    splits = [[]]
    for v in venues:
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, v['ask_size'])
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = None
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lam_o, lam_u, theta)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    if best_split is None:
        return [0] * len(venues), float('inf')
    return best_split, best_cost

def compute_cost(split, venues, order_size, lam_o, lam_u, theta):
    executed = 0
    cash_spent = 0
    for i, venue in enumerate(venues):
        exe = min(split[i], venue['ask_size'])
        executed += exe
        cash_spent += exe * (venue['ask'] + venue['fee'])
        maker_rebate = max(split[i] - exe, 0) * venue['rebate']
        cash_spent -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta * (underfill + overfill)
    cost_pen = lam_u * underfill + lam_o * overfill
    return cash_spent + risk_pen + cost_pen

def run_backtest(lambda_over, lambda_under, theta_queue):
    remaining = ORDER_SIZE
    total_cash = 0
    for _, snapshot in grouped:
        if remaining <= 0:
            break

        venues = []
        for _, row in snapshot.iterrows():
            venues.append({
                'ask': row['ask_px_00'],
                'ask_size': row['ask_sz_00'],
                'fee': 0.003,        
                'rebate': 0.001     
            })

        alloc, _ = allocate(min(remaining, ORDER_SIZE), venues,
                            lambda_over, lambda_under, theta_queue)

        for i, shares in enumerate(alloc):
            fill = min(shares, venues[i]['ask_size'])
            total_cash += fill * (venues[i]['ask'] + venues[i]['fee'])
            remaining -= fill
            if remaining <= 0:
                break

    avg_price = total_cash / ORDER_SIZE
    return total_cash, avg_price

def baseline_best_ask():
    remaining = ORDER_SIZE
    total_cash = 0
    for _, snapshot in grouped:
        if remaining <= 0:
            break
        best_row = snapshot.loc[snapshot['ask_px_00'].idxmin()]
        fill = min(remaining, best_row['ask_sz_00'])
        total_cash += fill * best_row['ask_px_00']
        remaining -= fill
    return total_cash, total_cash / ORDER_SIZE

def baseline_twap():
    total_cash = 0
    size_per_tick = ORDER_SIZE / len(grouped)
    for _, snapshot in grouped:
        prices = snapshot['ask_px_00']
        total_cash += prices.mean() * size_per_tick
    return total_cash, total_cash / ORDER_SIZE

def baseline_vwap():
    total_cash = 0
    total_volume = 0
    for _, snapshot in grouped:
        px_sz = snapshot['ask_px_00'] * snapshot['ask_sz_00']
        total_cash += px_sz.sum()
        total_volume += snapshot['ask_sz_00'].sum()
    vwap_price = total_cash / total_volume
    return vwap_price * ORDER_SIZE, vwap_price

results = []
best_result = {'cost': float('inf')}

for lam_o, lam_u, theta in PARAM_GRID:
    cost, avg = run_backtest(lam_o, lam_u, theta)
    results.append((lam_o, lam_u, theta, cost, avg))
    if cost < best_result['cost']:
        best_result = {
            'lambda_over': lam_o,
            'lambda_under': lam_u,
            'theta_queue': theta,
            'cost': cost,
            'avg_price': avg
        }

bb_cost, bb_avg = baseline_best_ask()
tw_cost, tw_avg = baseline_twap()
vw_cost, vw_avg = baseline_vwap()

json_output = {
    'best_parameters': best_result,
    'smart_order_router': {
        'total_cash': round(best_result['cost'], 2),
        'avg_fill_price': round(best_result['avg_price'], 4)
    },
    'baselines': {
        'best_ask': {'total_cash': round(bb_cost, 2), 'avg_fill_price': round(bb_avg, 4)},
        'twap': {'total_cash': round(tw_cost, 2), 'avg_fill_price': round(tw_avg, 4)},
        'vwap': {'total_cash': round(vw_cost, 2), 'avg_fill_price': round(vw_avg, 4)}
    },
    'savings_bps': {
        'vs_best_ask': round((1 - best_result['avg_price'] / bb_avg) * 10000, 2),
        'vs_twap': round((1 - best_result['avg_price'] / tw_avg) * 10000, 2),
        'vs_vwap': round((1 - best_result['avg_price'] / vw_avg) * 10000, 2)
    }
}

print(json.dumps(json_output, indent=4))

def generate_cumulative_cost_plot(lambda_over, lambda_under, theta_queue):
    remaining = ORDER_SIZE
    cumulative_costs = []
    total_cash = 0

    for _, snapshot in grouped:
        if remaining <= 0:
            break

        venues = []
        for _, row in snapshot.iterrows():
            venues.append({
                'ask': row['ask_px_00'],
                'ask_size': row['ask_sz_00'],
                'fee': 0.003,
                'rebate': 0.001
            })

        alloc, _ = allocate(min(remaining, ORDER_SIZE), venues,
                            lambda_over, lambda_under, theta_queue)

        for i, shares in enumerate(alloc):
            fill = min(shares, venues[i]['ask_size'])
            total_cash += fill * (venues[i]['ask'] + venues[i]['fee'])
            remaining -= fill
            cumulative_costs.append(total_cash)
            if remaining <= 0:
                break

    plt.figure(figsize=(7, 5))
    plt.plot(cumulative_costs)
    plt.title("Cumulative Cost: Optimal Allocator")
    plt.xlabel("Snapshot")
    plt.ylabel("Total Cost")
    plt.tight_layout()
    plt.savefig("results.png")  
    plt.close()

generate_cumulative_cost_plot(
    best_result['lambda_over'],
    best_result['lambda_under'],
    best_result['theta_queue']
)
