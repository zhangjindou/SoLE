# ----------------------- PATH ------------------------

ROOT_PATH = "."
DATA_PATH = "%s/../Datasets" % ROOT_PATH
FB15K_DATA_PATH = "%s/fb15k" % DATA_PATH
DB100K_DATA_PATH = "%s/db100k" % DATA_PATH
FB15K_SPARSE_DATA_PATH = "%s/fb15k-sparse" % DATA_PATH

LOG_PATH = "%s/log_dir" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH

# ----------------------- DATA ------------------------

DATASET = {}

FB15K_TRAIN_RAW = "%s/train.txt" % FB15K_DATA_PATH
FB15K_VALID_RAW = "%s/valid.txt" % FB15K_DATA_PATH
FB15K_TEST_RAW = "%s/test.txt" % FB15K_DATA_PATH
FB15K_TRAIN = "%s/digitized_train.txt" % FB15K_DATA_PATH
FB15K_VALID = "%s/digitized_valid.txt" % FB15K_DATA_PATH
FB15K_TEST = "%s/digitized_test.txt" % FB15K_DATA_PATH
FB15K_E2ID = "%s/e2id.txt" % FB15K_DATA_PATH
FB15K_R2ID = "%s/r2id.txt" % FB15K_DATA_PATH
FB15K_GNDS = "%s/groundings.txt" % FB15K_DATA_PATH
FB15K_RULES = "%s/lifted_rules.txt" % FB15K_DATA_PATH

DATASET["fb15k"] = {
    "train_raw": FB15K_TRAIN_RAW,
    "valid_raw": FB15K_VALID_RAW,
    "test_raw": FB15K_TEST_RAW,
    "train": FB15K_TRAIN,
    "valid": FB15K_VALID,
    "test": FB15K_TEST,
    "e2id": FB15K_E2ID,
    "r2id": FB15K_R2ID,
    "groundings": FB15K_GNDS,
}


DB100K_TRAIN_RAW = "%s/train.txt" % DB100K_DATA_PATH
DB100K_VALID_RAW = "%s/valid.txt" % DB100K_DATA_PATH
DB100K_TEST_RAW = "%s/test.txt" % DB100K_DATA_PATH
DB100K_TRAIN = "%s/digitized_train.txt" % DB100K_DATA_PATH
DB100K_VALID = "%s/digitized_valid.txt" % DB100K_DATA_PATH
DB100K_TEST = "%s/digitized_test.txt" % DB100K_DATA_PATH
DB100K_E2ID = "%s/e2id.txt" % DB100K_DATA_PATH
DB100K_R2ID = "%s/r2id.txt" % DB100K_DATA_PATH
DB100K_GNDS = "%s/groundings.txt" % DB100K_DATA_PATH

DATASET["db100k"] = {
    "train_raw": DB100K_TRAIN_RAW,
    "valid_raw": DB100K_VALID_RAW,
    "test_raw": DB100K_TEST_RAW,
    "train": DB100K_TRAIN,
    "valid": DB100K_VALID,
    "test": DB100K_TEST,
    "e2id": DB100K_E2ID,
    "r2id": DB100K_R2ID,
    "groundings": DB100K_GNDS,
}

FB15K_SPARSE_TRAIN_RAW = "%s/train.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_VALID_RAW = "%s/valid.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_TEST_RAW = "%s/test.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_TRAIN = "%s/digitized_train.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_VALID = "%s/digitized_valid.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_TEST = "%s/digitized_test.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_E2ID = "%s/e2id.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_R2ID = "%s/r2id.txt" % FB15K_SPARSE_DATA_PATH
FB15K_SPARSE_GNDS = "%s/groundings.txt" % FB15K_SPARSE_DATA_PATH

DATASET["fb15k-sparse"] = {
    "train_raw": FB15K_SPARSE_TRAIN_RAW,
    "valid_raw": FB15K_SPARSE_VALID_RAW,
    "test_raw": FB15K_SPARSE_TEST_RAW,
    "train": FB15K_SPARSE_TRAIN,
    "valid": FB15K_SPARSE_VALID,
    "test": FB15K_SPARSE_TEST,
    "e2id": FB15K_SPARSE_E2ID,
    "r2id": FB15K_SPARSE_R2ID,
    "groundings": FB15K_SPARSE_GNDS,
}

groundings = [str(50 + i * 5) for i in range(11)] + ['oneTime']
for item in groundings:
    DATASET["fb15k_" + str(item)] = {
        "train_raw": FB15K_TRAIN_RAW,
        "valid_raw": FB15K_VALID_RAW,
        "test_raw": FB15K_TEST_RAW,
        "train": FB15K_TRAIN,
        "valid": FB15K_VALID,
        "test": FB15K_TEST,
        "e2id": FB15K_E2ID,
        "r2id": FB15K_R2ID,
        "groundings": "%s/groundings_%s.txt" % (FB15K_DATA_PATH,str(item)),
    }
for item in groundings:
    DATASET["db100k_" + str(item)] = {
        "train_raw": DB100K_TRAIN_RAW,
        "valid_raw": DB100K_VALID_RAW,
        "test_raw": DB100K_TEST_RAW,
        "train": DB100K_TRAIN,
        "valid": DB100K_VALID,
        "test": DB100K_TEST,
        "e2id": DB100K_E2ID,
        "r2id": DB100K_R2ID,
        "groundings": "%s/groundings_%s.txt" % (DB100K_DATA_PATH,str(item)),
    }
# ----------------------- PARAM -----------------------

RANDOM_SEED = 123
