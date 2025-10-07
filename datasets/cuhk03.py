# encoding: utf-8
"""
CUHK03 Dataset
Reference:
Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
URL: https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html

Dataset statistics:
# identities: 1467
# images: ~13164 (labeled) + ~13164 (detected)
# cameras: 10 (5 pairs, 2 cameras per pair)
"""

import glob
import re
import os
import os.path as osp
from .bases import BaseImageDataset
from collections import defaultdict

class CUHK03(BaseImageDataset):
    """
    CUHK03 Dataset Class
    
    Dataset structure:
    data/CUHK03/
    ├── cuhk03_labeled/     # Labeled bounding boxes
    │   ├── 1_001_1_01.png
    │   ├── 1_001_2_06.png
    │   └── ...
    └── cuhk03_detected/    # Detected bounding boxes
        ├── 1_001_1_01.png
        ├── 1_001_2_06.png
        └── ...
    
    File naming: {pair}_{person}_{camera}_{frame}.png
    - pair: camera pair (1-5)
    - person: person index within pair
    - camera: camera within pair (1 or 2)
    - frame: frame number
    """
    dataset_dir = 'cuhk03'

    def __init__(self, root='', verbose=True, pid_begin=0, use_labeled=True, **kwargs):
        """
        Khởi tạo dataset CUHK03
        
        Args:
            root: Đường dẫn gốc chứa dataset (ví dụ: 'data/')
            verbose: Có in thông tin dataset không
            pid_begin: ID bắt đầu cho person ID (để tránh trùng lặp khi ghép nhiều dataset)
            use_labeled: True để dùng labeled bbox, False để dùng detected bbox
        """
        super(CUHK03, self).__init__()
        
        # Thiết lập đường dẫn đến dataset
        # self.dataset_dir = osp.join(root, self.dataset_dir)  # data/cuhk03
        self.dataset_dir = r"C:\Users\SpringT\Dropbox\MyWorld\Object_Tracking\TransReID\TransReID\data\cuhk03"  # data/cuhk03
        
        # Chọn thư mục labeled hoặc detected
        if use_labeled:
            self.data_dir = osp.join(self.dataset_dir, 'cuhk03_labeled')
            self.dataset_name = 'CUHK03-Labeled'
        else:
            self.data_dir = osp.join(self.dataset_dir, 'cuhk03_detected')
            self.dataset_name = 'CUHK03-Detected'

        # Kiểm tra xem thư mục có tồn tại không
        self._check_before_run()
        self.pid_begin = pid_begin
        
        # Xử lý dữ liệu - CUHK03 không có split sẵn nên cần tạo train/query/gallery
        all_data = self._process_dir(self.data_dir, relabel=True)

        # Chuẩn hóa camid toàn cục để liên tục từ 0..C-1 nhằm tương thích SIE
        # Lý do: SIE cần kích thước embedding theo số camera cố định; nếu train/test dùng dải cam khác nhau
        # thì sẽ dẫn đến lỗi shape/index khi test.
        all_camids = sorted({camid for _, _, camid, _ in all_data})
        camid2new = {camid: idx for idx, camid in enumerate(all_camids)}
        all_data = [(img, pid, camid2new[camid], viewid) for (img, pid, camid, viewid) in all_data]
        self.num_total_cams = len(all_camids)
        
        # Tạo split train/test cân bằng đảm bảo mỗi cam xuất hiện trong train
        train, query, gallery = self._create_splits(all_data)

        # In thông tin dataset nếu verbose=True
        if verbose:
            print("=> {} loaded".format(self.dataset_name))
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
        Kiểm tra xem thư mục cần thiết có tồn tại không
        Tránh lỗi khi xử lý dữ liệu nếu thiếu thư mục
        """
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Xử lý một thư mục chứa ảnh và trả về danh sách (img_path, person_id, camera_id, view_id)
        
        Args:
            dir_path: Đường dẫn đến thư mục chứa ảnh
            relabel: Có đánh số lại person ID không (True cho training, False cho test)
        
        Returns:
            List of tuples: [(img_path, person_id, camera_id, view_id), ...]
        """
        # Tìm tất cả file .png trong thư mục
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        
        # Định nghĩa pattern regex để parse tên file
        # Format: {pair}_{person}_{camera}_{frame}.png
        # Ví dụ: 1_001_1_01.png -> pair=1, person=001, camera=1, frame=01
        pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(\d+)\.png')

        # BƯỚC 1: Thu thập tất cả person ID từ tên file
        pid_container = set()  # Set để tránh trùng lặp person ID
        for img_path in sorted(img_paths):
            match = pattern.search(osp.basename(img_path))
            if match:
                pair, person, camera, frame = map(int, match.groups())
                # Tạo person ID duy nhất: pair * 1000 + person
                pid = pair * 1000 + person
                pid_container.add(pid)
        
        # Tạo mapping từ person ID gốc sang label mới (0, 1, 2, ...)
        # Ví dụ: {1001: 0, 1002: 1, 2001: 2, ...} nếu person_id gốc là 1001, 1002, 2001
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # BƯỚC 2: Xử lý từng ảnh và tạo dataset
        dataset = []
        for img_path in sorted(img_paths):
            # Parse tên file để lấy pair, person, camera, frame
            match = pattern.search(osp.basename(img_path))
            if not match:
                print(f"Warning: Cannot parse filename {img_path}")
                continue
                
            pair, person, camera, frame = map(int, match.groups())
            
            # Kiểm tra camera hợp lệ (1 hoặc 2)
            if camera not in [1, 2]:
                print(f"Warning: Invalid camera {camera} in {img_path}")
                continue
            
            # Tạo person ID duy nhất: pair * 1000 + person
            pid = pair * 1000 + person
            
            # Tạo camera ID toàn cục: (pair-1) * 2 + (camera-1)
            # Ví dụ: pair=1, camera=1 -> camid=0; pair=1, camera=2 -> camid=1
            # pair=2, camera=1 -> camid=2; pair=2, camera=2 -> camid=3
            camid = (pair - 1) * 2 + (camera - 1)
            
            # Đánh số lại person_id nếu cần (chỉ cho training set)
            if relabel: 
                pid = pid2label[pid]  # Chuyển person_id gốc thành label mới

            # Thêm vào dataset: (đường_dẫn_ảnh, person_id, camera_id, view_id)
            # view_id = frame (hoặc có thể dùng 1 nếu không cần)
            viewid = frame
            
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))
        
        return dataset

    def _create_splits(self, all_data):
        """
        Tạo split train/query/gallery cho CUHK03
        
        Args:
            all_data: Danh sách tất cả ảnh đã xử lý
            
        Returns:
            train, query, gallery: Các split của dataset
        """
        # Nhóm ảnh theo person ID
        person_images = defaultdict(list)
        for img_path, pid, camid, viewid in all_data:
            person_images[pid].append((img_path, pid, camid, viewid))
        
        # Lấy danh sách person ID và sắp xếp
        person_ids = sorted(person_images.keys())

        # Split theo tỷ lệ 767 train, 700 test (theo chuẩn CUHK03)
        # Hoặc có thể dùng tỷ lệ 70-30
        train_ratio = 0.52  # ~767/1467

        # Phân tầng theo camera: xác định quota theo camera và chọn pid để cân bằng cam
        # Mục tiêu: train có phân bố pid cân bằng theo từng cam, đồng thời giữ tổng số pid theo tỉ lệ train_ratio
        # Xây map cam -> danh sách pid có xuất hiện ở cam đó
        cam_to_pids = {}
        pid_to_cams = {}
        for pid in person_ids:
            cams = {camid for _, _, camid, _ in person_images[pid]}
            pid_to_cams[pid] = cams
            for cam in cams:
                cam_to_pids.setdefault(cam, []).append(pid)

        # Mục tiêu tổng số pid cho train (ít nhất bằng số camera để phủ đủ cam)
        total_train_target = int(len(person_ids) * train_ratio)
        total_train_target = max(total_train_target, len(cam_to_pids))

        # Quota cho từng camera (xấp xỉ theo tỷ lệ số pid có cam đó)
        cam_quota = {cam: int(len(set(cam_to_pids.get(cam, []))) * train_ratio) for cam in cam_to_pids.keys()}

        selected_train = set()

        # PASS 1: đảm bảo mỗi cam có ≥ 1 PID trong train
        for cam in sorted(cam_to_pids.keys()):
            if len(selected_train) >= total_train_target:
                break
            for pid in cam_to_pids[cam]:
                if pid in selected_train:
                    continue
                selected_train.add(pid)
                if cam_quota.get(cam, 0) > 0:
                    cam_quota[cam] -= 1
                break

        # PASS 2: phân bổ thêm theo quota, luân phiên qua các cam đến khi đạt tổng mục tiêu
        cams_sorted = sorted(cam_to_pids.keys())
        while len(selected_train) < total_train_target:
            progressed = False
            for cam in cams_sorted:
                if len(selected_train) >= total_train_target:
                    break
                if cam_quota.get(cam, 0) <= 0:
                    continue
                for pid in cam_to_pids[cam]:
                    if pid in selected_train:
                        continue
                    selected_train.add(pid)
                    cam_quota[cam] -= 1
                    progressed = True
                    break
            if not progressed:
                break

        # Nếu chưa đủ tổng target, bổ sung theo thứ tự pid còn lại (giữ tổng train ≈ train_ratio)
        if len(selected_train) < total_train_target:
            for pid in person_ids:
                if len(selected_train) >= total_train_target:
                    break
                if pid in selected_train:
                    continue
                selected_train.add(pid)

        train_pids = sorted(selected_train)
        test_pids = sorted(set(person_ids) - selected_train)
        
        # Tạo train set
        train = []
        for pid in train_pids:
            train.extend(person_images[pid])
        
        # Tạo query và gallery từ test set
        query = []
        gallery = []
        
        for pid in test_pids:
            images = person_images[pid]
            if len(images) > 0:
                # Lấy ảnh đầu tiên làm query
                query.append(images[0])
                # Các ảnh còn lại làm gallery
                gallery.extend(images[1:])
        
        return train, query, gallery
