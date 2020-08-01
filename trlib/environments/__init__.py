from .acrobot_multitask import AcrobotMultitask
from .dam import Dam
from .puddleworld import PuddleWorld
from gym.envs.registration import register
from .ags_base import TradingMain
from .ags_derivatives import TradingDerivatives
from .ags_prices import TradingPrices


register(
    id='Dam-v0',
    entry_point='trlib.environments.dam:Dam',
)

register(
    id='AgsPrices-v0',
    entry_point='trlib.environments.ags_prices:TradingPrices',
)

register(
    id='AgsDerivatives-v0',
    entry_point='trlib.environments.ags_derivatives:TradingDerivatives',
)

register(
    id='CarOnHill-v0',
    entry_point='trlib.environments.car_on_hill:CarOnHill',
)

register(
    id='Pendulum-v1',
    entry_point='trlib.environments.pendulum:Pendulum',
)

register(
    id='AcrobotMultiTask-v0',
    entry_point='trlib.environments.acrobot_multitask:AcrobotMultitask',
)

register(
    id='Trading2017-EURUSDJPY-v1',
    entry_point='trlib.environments.trading_proportional:Trading',
    kwargs={
        'csv_path': '2017-EURUSDJPY.csv',
        'data_name1':'EURUSD',
        'data_name2':'USDJPY',
        'fees': 1e-6
    }
)

register(
    id='Trading2017_EURUSD-v1',
    entry_point='trlib.environments.trading_proportional:Trading',
    kwargs={
        'csv_path': '2017-EURUSDJPY.csv',
        'data_name1':'EURUSD',
        'data_name2':'None',
        'fees': 1e-6
    }
)


register(
    id='Old-Trading2017_EURUSD_Week-v1',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': '2017-EURUSD-BGN-Curncy-1m.csv',
    }
)

register(
    id='Old-Trading2017_EURUSD-v1',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': '2017-EURUSD-BGN-Curncy-1m.csv',
    }
)
register(
    id='Old-Trading2017_EURUSD-v2',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': '2017-EURUSD-BGN-Curncy-1m.csv',
        'fees': 3e-5
    }
)

register(
    id='Old-Trading2017_EURUSD-v3',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': '2017-EURUSD-BGN-Curncy-1m.csv',
        'fees': 4e-5
    }
)

register(
    id='Old-Trading2017_EURUSD-v4',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': '2017-EURUSD-BGN-Curncy-1m.csv',
        'fees': 5e-5
    }
)

register(
    id='Old-Trading2017_EURUSD-v5',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': 'EURUSD_2017_1_over_8.csv',
        'fees': 2e-5
    }
)

register(
    id='Old-Trading2017_EURUSD-v6',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': 'EURUSD_2017_1_over_8.csv',
        'fees': 3e-5
    }
)

register(
    id='Old-Trading2017_EURUSD-v7',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': 'EURUSD_2017_1_over_8.csv',
        'fees': 4e-5
    }
)

register(
    id='Old-Trading2017_EURUSD-v8',
    entry_point='trlib.environments.vec_prices:VecTradingPrices',
    kwargs={
        'data': 'EURUSD_2017_1_over_8.csv',
        'fees': 5e-5
    }
)



register(
    id='Gridworld-v1',
    entry_point='trlib.environments.gridworld_continuous:GridWorldContinuous',
)

register(
    id='Gridworld-v2',
    entry_point='trlib.environments.gridworld_continuous2:GridWorldContinuous',
)

register(
    id='Gridworld-v3',
    entry_point='trlib.environments.gridworld_continuous3:GridWorldContinuous',
)