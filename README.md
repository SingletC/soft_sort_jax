# Soft Sort JAX
AD support for soft sort
```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import grad
from soft_sort_jax.jax_ops import soft_sort
x = jnp.array([1., 2., 3.], dtype=jnp.float64)
def test(x):
    sorted = soft_sort(x.reshape(1,-1), regularization_strength=0.1)[0]
    return sorted[0]*sorted[1]* sorted[2]
print(test(x))
# 6.0
print(jax.grad(test)(x))
# [6. 3. 2.]
print(jax.hessian(test)(x))        
#[[0.         2.99999997 1.99999998]
# [2.99999997 0.         0.99999999]
# [1.99999998 0.99999999 0.        ]]
```
Note performance is very bad. Some code need to be optimized.  
```
%timeit jax.hessian(test)(x)
#73.5 ms ± 162 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
and jit does not work yet.
