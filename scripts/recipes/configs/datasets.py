# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"
    
@dataclass
class Irishman_dataset_REMI:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/REMI/small_train.json"
    test_data_path: str = "../../dataset/Irishman/REMI/small_validation.json"
    input_length: int = 2048
    # train_data_path: str = "../../dataset/Irishman/REMI/train.json"    
    # test_data_path: str = "../../dataset/Irishman/REMI/validation.json"

class Irishman_dataset_MIDILike:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/MIDILike/small_train.json"
    test_data_path: str = "../../dataset/Irishman/MIDILike/small_validation.json"
    input_length: int = 2048

class Irishman_dataset_TSD:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/TSD/small_train.json"
    test_data_path: str = "../../dataset/Irishman/TSD/small_validation.json"
    input_length: int = 2048

class Irishman_dataset_Structured:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/Structured/small_train.json"
    test_data_path: str = "../../dataset/Irishman/Structured/small_validation.json"
    input_length: int = 2048

class Irishman_dataset_CPWord:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/CPWord/small_train.json"
    test_data_path: str = "../../dataset/Irishman/CPWord/small_validation.json"
    input_length: int = 2048

class Irishman_dataset_Octuple:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/Octuple/small_train.json"
    test_data_path: str = "../../dataset/Irishman/Octuple/small_validation.json"
    input_length: int = 2048

class Irishman_dataset_MuMIDI:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/MuMIDI/small_train.json"
    test_data_path: str = "../../dataset/Irishman/MuMIDI/small_validation.json"
    input_length: int = 2048

class Irishman_dataset_MMM:
    dataset: str = "Irishman_dataset"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = "../../dataset/Irishman/MMM/small_train.json"
    test_data_path: str = "../../dataset/Irishman/MMM/small_validation.json"
    input_length: int = 2048

