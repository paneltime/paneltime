#!/usr/bin/env python
# -*- coding: utf-8 -*-

def save_csv(fname, array, sep = ','):
  f = open(fname, 'wt')
  for line in array:
    f.write(sep.join(line))
  f.close()
