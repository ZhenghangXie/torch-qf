"""Functions that get the price and greeks of Vanilla options based on Black-Scholes"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from scipy.stats import norm

pi = 3.14159265359

def get_vanilla_prices(*,
                     strikes,
                     volatilities,
                     expiries,
                     spots=None,
                     forwards=None,
                     discount_rates=None,
                     continuous_dividends=None,
                     cost_of_carries=None,
                     discount_factors=None,
                     is_call_options=True):
    '''
    Computes the Black Scholes price for a batch of call or put options.
    '''
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    if (continuous_dividends is not None) and (cost_of_carries is not None):
        raise ValueError('At most one of continuous_dividends and cost_of_carries may be supplied')

    t_strikes = torch.Tensor(strikes)
    t_volatilities = torch.Tensor(volatilities)
    t_expiries = torch.Tensor(expiries)

    if discount_rates is not None:
        t_discount_rates = torch.Tensor(discount_rates)
    elif discount_factors is not None:
        t_discount_rates = -torch.log(discount_factors) / expiries
    else:
        t_discount_rates = torch.zeros_like(t_volatilities)

    if continuous_dividends is None:
        t_continuous_dividends = torch.zeros_like(t_volatilities)

    if cost_of_carries is not None:
        t_cost_of_carries = torch.Tensor(cost_of_carries)
    else:
        t_cost_of_carries = t_discount_rates - t_continuous_dividends

    if discount_factors is not None:
        t_discount_factors = torch.Tensor(discount_factors)
    else:
        t_discount_factors = torch.exp(-t_discount_rates * t_expiries)

    if forwards is not None:
        t_forwards = torch.Tensor(forwards)
        t_spots * torch.exp(-t_cost_of_carries * t_expiries)
    else:
        t_spots = torch.Tensor(spots)
        t_forwards = t_spots * torch.exp(t_cost_of_carries * t_expiries)

    t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
    d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
    d2 = d1 - t_sqrt_var
    t_undiscounted_calls = t_forwards * torch.Tensor(norm.cdf(d1)) - t_strikes * torch.Tensor(norm.cdf(d2))
    if is_call_options == True:
        return t_discount_factors * t_undiscounted_calls
    else:
        t_undiscounted_forward = t_forwards - t_strikes
        t_undiscounted_puts = t_undiscounted_calls - t_undiscounted_forward
        return discount_factors * t_undiscounted_puts


def get_vanilla_greeks(*,
                       strikes,
                       volatilities,
                       expiries,
                       greek,
                       spots=None,
                       forwards=None,
                       discount_rates=None,
                       continuous_dividends=None,
                       cost_of_carries=None,
                       discount_factors=None,
                       is_call_options=True):
    '''
    Computes the Greeks of a batch of call or put plain vanilla options.
    '''
    if greek not in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        raise ValueError('Input greek should be one of \'delta\',\'gamma\',\'theta\',\'vega\',\'rho\'')
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if (discount_rates is not None) and (discount_factors is not None):
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    if (continuous_dividends is not None) and (cost_of_carries is not None):
        raise ValueError('At most one of continuous_dividends and cost_of_carries may be supplied')

    t_strikes = torch.Tensor(strikes)
    t_volatilities = torch.Tensor(volatilities)
    t_expiries = torch.Tensor(expiries)

    if discount_rates is not None:
        t_discount_rates = torch.Tensor(discount_rates)
    elif discount_factors is not None:
        t_discount_rates = -torch.log(discount_factors) / expiries
    else:
        t_discount_rates = torch.zeros_like(t_volatilities)

    if continuous_dividends is None:
        t_continuous_dividends = torch.zeros_like(t_volatilities)

    if cost_of_carries is not None:
        t_cost_of_carries = torch.Tensor(cost_of_carries)
    else:
        t_cost_of_carries = t_discount_rates - t_continuous_dividends

    if discount_factors is not None:
        t_discount_factors = torch.Tensor(discount_factors)
    else:
        t_discount_factors = torch.exp(-t_discount_rates * t_expiries)

    if forwards is not None:
        t_forwards = torch.Tensor(forwards)
        t_spots = t_forwards * torch.exp(-t_cost_of_carries * t_expiries)
    else:
        t_spots = torch.Tensor(spots)
        t_forwards = t_spots * torch.exp(t_cost_of_carries * t_expiries)

    if is_call_options == True:
        if greek == 'delta':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            delta = torch.Tensor(norm.cdf(d1))
            return delta
        if greek == 'gamma':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            gamma = (torch.exp(-d1 ** 2 / 2)) / (torch.sqrt(2 * pi * (t_expiries)) * t_spots * t_volatilities)
            return gamma
        if greek == 'theta':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            d2 = d1 - t_sqrt_var
            theta = t_spots * t_volatilities * (torch.exp(-d1 ** 2 / 2)) / (torch.sqrt(2 * pi * (t_expiries))) - \
                    t_cost_of_carries * t_strikes * torch.exp(-t_cost_of_carries * t_expiries) * torch.Tensor(
                norm.cdf(d2))
            return theta
        if greek == 'vega':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            vega = t_spots * torch.sqrt(t_expiries) * torch.exp(-d1 ** 2 / 2)
            return vega
        if greek == 'rho':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            d2 = d1 - t_sqrt_var
            rho = t_strikes * t_expiries * torch.exp(-t_cost_of_carries * t_expiries) * torch.Tensor(norm.cdf(d2))
            return rho

    if is_call_options == False:
        if greek == 'delta':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            delta = torch.Tensor(norm.cdf(d1)) - 1
            return delta
        if greek == 'gamma':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            gamma = (torch.exp(-d1 ** 2 / 2)) / (torch.sqrt(2 * pi * (t_expiries)) * t_spots * t_volatilities)
            return gamma
        if greek == 'theta':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            d2 = d1 - t_sqrt_var
            theta = -t_spots * t_volatilities * (torch.exp(-d1 ** 2 / 2)) / (torch.sqrt(2 * pi * (t_expiries))) + \
                    t_cost_of_carries * t_strikes * torch.exp(-t_cost_of_carries * t_expiries) * torch.Tensor(
                norm.cdf(-d2))
            return theta
        if greek == 'vega':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            vega = t_spots * torch.sqrt(t_expiries) * torch.exp(-d1 ** 2 / 2)
            return vega
        if greek == 'rho':
            t_sqrt_var = t_volatilities * torch.sqrt(t_expiries)
            d1 = (torch.log(t_forwards / t_strikes) + t_sqrt_var * t_sqrt_var / 2) / t_sqrt_var
            d2 = d1 - t_sqrt_var
            rho = -t_strikes * t_expiries * torch.exp(-t_cost_of_carries * t_expiries) * torch.Tensor(norm.cdf(-d2))
            return rho
    else:
        raise ValueError('Variable \'is_call_options\' can only be Booleans ')

