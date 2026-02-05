from jax import config

PRECISION = 'double'  # 'double'

DO_JIT = True

DEBUG = False


def update_jax_precision(precision):
    if precision == 'double':
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)


update_jax_precision(PRECISION)

__all__ = ['PRECISION', 'DO_JIT', 'DEBUG', "update_jax_precision"]
