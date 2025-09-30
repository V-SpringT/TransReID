# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob  # Tìm kiếm file theo pattern (wildcard)
import re    # Xử lý regular expression để parse tên file

import os.path as osp  # Xử lý đường dẫn file cross-platform

from .bases import BaseImageDataset  # Class cơ sở cho tất cả dataset
from collections import defaultdict  # Dictionary với giá trị mặc định
import pickle  # Serialize/deserialize Python objects
class Market1501(BaseImageDataset):
    """
    Market1501 Dataset Class
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background) - Số lượng người trong dataset
    # images: 12936 (train) + 3368 (query) + 15913 (gallery) - Tổng số ảnh
    """
    dataset_dir = 'market1501'  # Tên thư mục chứa dataset

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        """
        Khởi tạo dataset Market1501
        
        Args:
            root: Đường dẫn gốc chứa dataset (ví dụ: 'data/')
            verbose: Có in thông tin dataset không
            pid_begin: ID bắt đầu cho person ID (để tránh trùng lặp khi ghép nhiều dataset)
        """
        super(Market1501, self).__init__()
        
        # Thiết lập đường dẫn đến các thư mục con
        self.dataset_dir = osp.join(root, self.dataset_dir)  # data/market1501
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')  # data/market1501/bounding_box_train
        self.query_dir = osp.join(self.dataset_dir, 'query')  # data/market1501/query
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')  # data/market1501/bounding_box_test

        # Kiểm tra xem tất cả thư mục có tồn tại không
        self._check_before_run()
        self.pid_begin = pid_begin
        
        # Xử lý dữ liệu từ các thư mục
        train = self._process_dir(self.train_dir, relabel=True)    # Xử lý ảnh training (relabel=True để đánh số lại person ID)
        query = self._process_dir(self.query_dir, relabel=False)  # Xử lý ảnh query (relabel=False giữ nguyên person ID)
        gallery = self._process_dir(self.gallery_dir, relabel=False)  # Xử lý ảnh gallery (relabel=False giữ nguyên person ID)

        # In thông tin dataset nếu verbose=True
        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        # Lưu dữ liệu đã xử lý
        self.train = train      # Danh sách ảnh training
        self.query = query      # Danh sách ảnh query (để test)
        self.gallery = gallery  # Danh sách ảnh gallery (để so sánh với query)

        # Tính toán thống kê dataset
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """
        Kiểm tra xem tất cả thư mục cần thiết có tồn tại không
        Tránh lỗi khi xử lý dữ liệu nếu thiếu thư mục
        """
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Xử lý một thư mục chứa ảnh và trả về danh sách (img_path, person_id, camera_id, view_id)
        
        Args:
            dir_path: Đường dẫn đến thư mục chứa ảnh
            relabel: Có đánh số lại person ID không (True cho training, False cho test)
        
        Returns:
            List of tuples: [(img_path, person_id, camera_id, view_id), ...]
        """
        # Tìm tất cả file .jpg trong thư mục
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        # Định nghĩa pattern regex để parse tên file
        # Format: {person_id}_c{camera_id}s{sequence}_{frame}_{bbox}.jpg
        # Ví dụ: 0002_c1s1_000451_03.jpg -> person_id=0002, camera_id=1
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # BƯỚC 1: Thu thập tất cả person ID từ tên file
        pid_container = set()  # Set để tránh trùng lặp person ID
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())  # Parse person_id và camera_id
            if pid == -1: continue  # Bỏ qua ảnh junk (có person_id = -1)
            pid_container.add(pid)
        
        # Tạo mapping từ person ID gốc sang label mới (0, 1, 2, ...)
        # Ví dụ: {2: 0, 3: 1, 5: 2, ...} nếu person_id gốc là 2, 3, 5
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # BƯỚC 2: Xử lý từng ảnh và tạo dataset
        dataset = []
        for img_path in sorted(img_paths):
            # Parse tên file để lấy person_id và camera_id
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # Bỏ qua ảnh junk
            
            # Kiểm tra person_id hợp lệ (0-1501, trong đó 0 là background)
            assert 0 <= pid <= 1501  # pid == 0 means background
            # Kiểm tra camera_id hợp lệ (1-6)
            assert 1 <= camid <= 6
            
            # Chuyển camera_id từ 1-6 thành 0-5 (0-indexed)
            camid -= 1  # index starts from 0
            
            # Đánh số lại person_id nếu cần (chỉ cho training set)
            if relabel: 
                pid = pid2label[pid]  # Chuyển person_id gốc thành label mới

            # Thêm vào dataset: (đường_dẫn_ảnh, person_id, camera_id, view_id)
            # view_id = 1 (góc nhìn, thường là 1 cho Market1501)
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        
        return dataset
