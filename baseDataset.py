from collections import defaultdict
from torch.utils.data import Dataset
import os


class baseDataset(object):
    def __init__(self, trainpath, testpath, statpath, validpath):
        """base Dataset. Read data files and preprocess.
        Args:
            trainpath: File path of train Data;
            testpath: File path of test data;
            statpath: File path of entities num and relatioins num;
            validpath: File path of valid data
        """
        # 加载训练、测试和验证四元组
        self.trainQuadruples = self.load_quadruples(trainpath)
        self.testQuadruples = self.load_quadruples(testpath)
        self.validQuadruples = self.load_quadruples(validpath)
        # 合并所有四元组
        self.allQuadruples = self.trainQuadruples + self.validQuadruples + self.testQuadruples
        # 获取实体和关系的总数
        self.num_e, self.num_r = self.get_total_number(statpath)  # number of entities, number of relations
        # 它根据输入的四元组数据，生成一个映射，键为 (实体, 关系, 时间戳)，值为与该键相关的真实实体的集合
        self.skip_dict = self.get_skipdict(self.allQuadruples)

        # 存储训练集中出现的实体
        self.train_entities = set()  # Entities that have appeared in the training set
        for query in self.trainQuadruples:
            self.train_entities.add(query[0])
            self.train_entities.add(query[2])

        # ADDED: Create sets of (h, r, t) triplets for checking if a query is new or seen.
        # These sets ignore the timestamp.
        self.train_triplets = {(h, r, o) for h, r, o, _ in self.trainQuadruples}
        self.train_valid_triplets = {(h, r, o) for h, r, o, _ in self.trainQuadruples + self.validQuadruples}

        # ADDED: Load entity and relation text mappings
        data_dir = os.path.dirname(trainpath)
        self.id2entity = self._load_mappings(os.path.join(data_dir, 'entity2id.txt'))
        self.id2relation = self._load_mappings(os.path.join(data_dir, 'relation2id.txt'))

    # ADDED: Helper function to load ID to text mappings
    @staticmethod
    def _load_mappings(filepath):
        """Loads ID to text mappings from a file.
        File format: text_description\t id
        Returns:
            A dictionary mapping from ID (int) to text (str).
        """
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    # The format is typically name then id
                    name, entity_id = parts[0], int(parts[1])
                    mapping[entity_id] = name
        return mapping

    # 获取数据集中的所有时间戳
    def get_all_timestamps(self):
        """Get all the timestamps in the dataset.
        return:
            timestamps: a set of timestamps.
        """
        timestamps = set()
        for ex in self.allQuadruples:
            timestamps.add(ex[3])
        return timestamps

    def get_skipdict(self, quadruples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for src, rel, dst, time in quadruples:
            filters[(src, rel, time)].add(dst)
            # filters[(dst, rel+self.num_r+1, time)].add(src)
        return filters

    @staticmethod
    def load_quadruples(inpath):
        """train.txt/valid.txt/test.txt reader
        inpath: File path. train.txt, valid.txt or test.txt of a dataset;
        return:
            quadrupleList: A list
            containing all quadruples([subject/headEntity, relation, object/tailEntity, timestamp]) in the file.
        """
        with open(inpath, 'r') as f:
            quadrupleList = []
            for line in f:
                try:
                    line_split = line.split()
                    head = int(line_split[0])
                    rel = int(line_split[1])
                    tail = int(line_split[2])
                    time = int(line_split[3])
                    quadrupleList.append([head, rel, tail, time])
                except:
                    print(line)
        return quadrupleList

    @staticmethod
    def get_total_number(statpath):
        """stat.txt reader
        return:
            (number of entities -> int, number of relations -> int)
        """
        with open(statpath, 'r') as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])


# QuadruplesDataset 类是一个继承自 PyTorch 的 Dataset 类，用于存储和管理四元组数据。
# 这个类支持数据的访问和迭代，方便在深度学习模型中使用。
class QuadruplesDataset(Dataset):
    def __init__(self, examples, num_r):
        """
        examples: a list of quadruples.
        num_r: number of relations
        """
        self.quadruples = examples.copy()
        # for ex in examples:
        #     self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]])

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, item):
        return self.quadruples[item][0], \
            self.quadruples[item][1], \
            self.quadruples[item][2], \
            self.quadruples[item][3]
