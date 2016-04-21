#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

def kwargs_update(default,**kwargs):
    """ This is native in python 3 **default + **kwargs """
    k = default.copy()
    for key,val in kwargs.iteritems():
        k[key] = val
        
    return k
