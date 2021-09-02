import numpy as np

import joblib
import torch

import config
import dataset
import engine
import utils
from model import EntityModel


if __name__ == "__main__":

    sentence = "background information ( 09/2019 - 12/2019 ) restaurant management " \
               "system ( console application ) education da nang university science " \
               "technology 09/2018 - present major : information technology working " \
               "experience position : c + + programming - design database . payments " \
               ". - coded functions : add food , drinks , menu ... ; order meal , " \
               "tutors management system da nang ( desktop application ) ( 03/2020 " \
               "- 07/2020 ) position : c # ( . net ) programming - design interface " \
               ". hoang nguyen vu interns 12/04 / 2000 male 03398784 81 " \
               "hoangnguyenvubk@gmail.com 850a ton duc thang , lien chieu " \
               ", da nang facebook.com/nguyenvu124 career goals - coded gui " \
               "layer half bus layer three-layers model . hope work professional " \
               "programming environment , develop professional skills , learn new " \
               "programming knowledge successfully complete assigned work . atm " \
               "system simulation program banks using multi - access ( web " \
               "application ) ( 09/2020 - present ) position : java ( jsp / " \
               "servlet ) programming - design database , interface personal " \
               "skills - coded controller , model.bo , model.bean mvc model . c " \
               "+ + language c # ( . net ) java ( jsp / servlet ) skills - " \
               "working database microsoft sql mysql . - c + + , java , c # " \
               ". - html little skills css . - javascript . - nodejs ( learning ) ."
    result = utils.predict(sentence)
