import numpy as np
import time
import matplotlib.pyplot as plt

def engine():

    print("-" * 50)


    points = 10_000_000
    i_price = 100.0
    
    start_bench = time.perf_counter()
    

    price_sh = np.random.standard_normal(points)
    
    prices = np.cumsum(price_sh) + i_price
    
    print(f"10,000,000 ticks generated.")


    
    def vec_sma(data, win):
      
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[win:] - cumsum[:-win]) / win

   
    ma50 = vec_sma(prices, 50)
    ma200 = vec_sma(prices, 200)

   
    ma50_al = ma50[len(ma50) - len(ma200):]
    
  
    signals = (ma50_al > ma200).astype(int)

  
    r_prices = prices[-len(signals):]
    price_d = np.diff(r_prices)
    

    pnl_vec = signals[:-1] * price_d
    
    t_pnl = np.sum(pnl_vec)
    
    end_bench = time.perf_counter()
    t_time = end_bench - start_bench


    status = "Profit" if t_pnl > 0 else "Loss"
    
    print(f" Signals: {len(signals)} signals processed.")
    print(f"Result: {status}")
    print(f"PnL: {t_pnl:,.2f}")
    print(f"Total Time: {t_time:.4f} seconds")
    print("-" * 50)

    

engine()