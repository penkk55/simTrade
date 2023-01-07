# # This is a sample Python script.
# import numpy  as np
#
#
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#     print(np)
#

# 1.Create a simulator with custom parameters

# import pytz
# from datetime import datetime, timedelta
# from gym_mtsim import MtSimulator, OrderType, Timeframe, FOREX_DATA_PATH
#
#
# sim = MtSimulator(
#     unit='USD',
#     balance=10000.,
#     leverage=100.,
#     stop_out_level=0.2,
#     hedge=False,
# )
#
# if not sim.load_symbols(FOREX_DATA_PATH):
#     sim.download_data(
#         symbols=['EURUSD', 'GBPCAD', 'GBPUSD', 'USDCAD', 'USDCHF', 'GBPJPY', 'USDJPY'],
#         time_range=(
#             datetime(2021, 5, 5, tzinfo=pytz.UTC),
#             datetime(2021, 9, 5, tzinfo=pytz.UTC)
#         ),
#         timeframe=Timeframe.D1
#     )
#     sim.save_symbols(FOREX_DATA_PATH)
#
#
# # 2. Place some orders
# sim.current_time = datetime(2021, 8, 30, 0, 17, 52, tzinfo=pytz.UTC)
#
# order1 = sim.create_order(
#     order_type=OrderType.Buy,
#     symbol='GBPCAD',
#     volume=1.,
#     fee=0.0003,
# )
#
# sim.tick(timedelta(days=2))
#
# order2 = sim.create_order(
#     order_type=OrderType.Sell,
#     symbol='USDJPY',
#     volume=2.,
#     fee=0.01,
# )
#
# sim.tick(timedelta(days=5))
#
# state = sim.get_state()
#
# print(
#     f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
#     f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
# )
# state['orders']
#
#
# order1_profit = sim.close_order(order1)
# order2_profit = sim.close_order(order2)
#
# # alternatively:
# # for order in sim.orders:
# #     sim.close_order(order)
#
# state = sim.get_state()
#
# print(
#     f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
#     f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
# )
# state['orders']



# ex
import gym
from gym_mtsim import (
    Timeframe, SymbolInfo,
    MtSimulator, OrderType, Order, SymbolNotFound, OrderNotFound,
    MtEnv,
    FOREX_DATA_PATH, STOCKS_DATA_PATH, CRYPTO_DATA_PATH, MIXED_DATA_PATH,
)
env = gym.make('forex-hedge-v0')
# env = gym.make('stocks-hedge-v0')
# env = gym.make('crypto-hedge-v0')
# env = gym.make('mixed-hedge-v0')

# env = gym.make('forex-unhedge-v0')
# env = gym.make('stocks-unhedge-v0')
# env = gym.make('crypto-unhedge-v0')
# env = gym.make('mixed-unhedge-v0')
import pytz
from datetime import datetime, timedelta
import numpy as np
from gym_mtsim import MtEnv, MtSimulator, FOREX_DATA_PATH


sim = MtSimulator(
    unit='USD',
    balance=10000.,
    leverage=100.,
    stop_out_level=0.2,
    hedge=True,
    symbols_filename=FOREX_DATA_PATH
)

env = MtEnv(
    original_simulator=sim,
    trading_symbols=['GBPCAD', 'EURUSD', 'USDJPY'],
    window_size=10,
    # time_points=[desired time points ...],
    hold_threshold=0.5,
    close_threshold=0.5,
    fee=lambda symbol: {
        'GBPCAD': max(0., np.random.normal(0.0007, 0.00005)),
        'EURUSD': max(0., np.random.normal(0.0002, 0.00003)),
        'USDJPY': max(0., np.random.normal(0.02, 0.003)),
    }[symbol],
    symbol_max_orders=2,
    multiprocessing_processes=2
)
print("env information:")

for symbol in env.prices:
    print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)

print("> signal_features.shape:", env.signal_features.shape)
print("> features_shape:", env.features_shape)



from stable_baselines3 import A2C


# env = gym.make('forex-hedge-v0')

model = A2C('MultiInputPolicy', env, verbose=0)
model.learn(total_timesteps=1000)

observation = env.reset()
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break

env.render('advanced_figure', time_format="%Y-%m-%d")
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # print_hi('PyCharm')

    print('pycharm_mtsim')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
