#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

def kwargs_update(default,**kwargs):
    """ This is native in python 3 **default + **kwargs """
    k = default.copy()
    for key,val in kwargs.iteritems():
        k[key] = val
        
    return k

def make_method(obj):
    """Decorator to make the function a method of *obj*.

    In the current context::
      @make_method(Axes)
      def toto(ax, ...):
          ...
    makes *toto* a method of `Axes`, so that one can directly use::
      ax.toto()
    COPYRIGHT: from Yannick Copin
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f

    return decorate
