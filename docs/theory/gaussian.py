import numpy as np
 
def gaussian_pdf(x, mu=0.0, sigma=1.0):
    """N(x; mu, sigma^2) probability density."""
    x = np.asarray(x, dtype=float) 
    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi)) #this is the normalization coefficient , without this the area under the "bell curve" wont be equal to 1
    return coeff * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

""" pdf --> probability density function , this function basically gives uses the famous gaussian bell curve formula to calculate basically how likely a value is near x 
it doesnt give the probability rather gives a function which will give us the probability function"""

#this is the erf (error function),
def _erf(x):
    """Abramowitz & Stegun 7.1.26 rational approximation, |error| < 1.5e-7."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y

""" basically as the gaussian cant be just calculated using elementary calculus so we came up with a formula but here we are using approximations which are the coeff a1,a2 etc  """

#cd is cumulative distribution function , basically asks the probability of the input being less than x
def gaussian_cdf(x, mu=0.0, sigma=1.0):
    """P(X <= x) for X ~ N(mu, sigma^2), via the error function."""
    z = (np.asarray(x, dtype=float) - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))

""" uses the simple forula and the erf function to get the answr  """

def sample_uniform(n, low=0.0, high=1.0, rng=None):
    rng = rng or np.random.default_rng()
    return rng.uniform(low, high, size=n)
 
""" every number has equal probability  """

def sample_gaussian_box_muller(n, mu=0.0, sigma=1.0, rng=None):
    """Box-Muller transform: turns two uniforms into two independent Gaussians."""
    rng = rng or np.random.default_rng()
    n_pairs = int(np.ceil(n / 2))
    u1 = rng.uniform(size=n_pairs)
    u2 = rng.uniform(size=n_pairs)
    r = np.sqrt(-2.0 * np.log(u1))
    z0 = r * np.cos(2 * np.pi * u2)
    z1 = r * np.sin(2 * np.pi * u2)
    z = np.concatenate([z0, z1])[:n]
    return mu + sigma * z

""" okay so we first generate unifrom then we convert the uniform random number to gaussian random  """

def sample_inverse_cdf(n, cdf_inv, rng=None):
    """Generic inverse-transform sampling given the inverse CDF function."""
    rng = rng or np.random.default_rng()
    u = rng.uniform(size=n)
    return cdf_inv(u)
 
"""  """

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    samples = sample_gaussian_box_muller(100_000, mu=0.0, sigma=1.0, rng=rng)
    print("mean:", samples.mean(), "std:", samples.std())
    print("CDF(0) ~ 0.5:", gaussian_cdf(0.0))
