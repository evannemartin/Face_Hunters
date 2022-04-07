def sum(a,b):
    """
    return the sum of the 2 input a and b.

    Args :
        a (float): nb1\n
        b (float) : nb2

    Returns :
        float : sum

    Examples :
        >>> sum(1,3)
        4
        >>> sum(40,10)
        50

    """

    return a+b

if __name__=="__main__" :
    print(sum(2,3))
    import doctest
    doctest.testmod(verbose=True)

#sphinx-apidoc -f -o source/  Project/
#deploy to test PyPi
