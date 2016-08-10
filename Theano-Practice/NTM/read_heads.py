import util


def reading(memory_t, weight_t):
    """
    Reading memory.
    :param memory_t: the $N \times M$ memory matrix at time $t$, where $N$
    is the number of memory locations, and $M$ is the vector size at each
    location.
    :param weight_t: $w_t$ is a vector of weightings over the $N$ locations
    emitted by a reading head at time $t$.

    Since all weightings are normalized, the $N$ elements $w_t(i)$ of
    $\textbf{w}_t$ obey the following constraints:
    $$\sum_{i=1}^{N} w_t(i) = 1, 0 \le w_t(i) \le 1,\forall i$$

    The length $M$ read vector $r_t$ returned by the head is defined as a
    convex combination of the row-vectors $M_t(i)$ in memory:
    $$\textbf{r}_t \leftarrow \sum_{i=1}^{N}w_t(i)\textbf{M}_t(i)$$
    :return: the content reading from memory.
    """
    r_t = memory_t * weight_t
    return r_t
