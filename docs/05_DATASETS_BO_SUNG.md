# Dataset bổ sung cho nhận diện visual style

Cập nhật: 2026-10-17. Đây là danh sách tải và kế hoạch sử dụng; pipeline hiện tại chưa tự đọc hoặc train trên ba nguồn mới này.

## Thứ tự tải

| Ưu tiên | Dataset | Dùng trong dự án | Nguồn chính thức |
| --- | --- | --- | --- |
| 1 | DeepFashion — Category and Attribute Prediction | Học thuộc tính trang phục và kiểm tra đặc trưng ảnh chi tiết hơn loại sản phẩm/màu | [Trang benchmark](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) |
| 2 | Fashionpedia | Học thuộc tính ở từng món đồ, dùng mask để giảm ảnh hưởng của nền và người mẫu | [Trang dự án](https://fashionpedia.github.io/home/), [Download](https://github.com/cvdfoundation/fashionpedia#download) |
| 3 | DeepFashion2 | Học cùng một món đồ qua ảnh shop và ảnh người dùng; đánh giá khả năng nhận diện qua nguồn ảnh khác nhau | [Tài liệu và Download](https://github.com/switchablenorms/DeepFashion2#download-the-data) |

Đây là đề xuất sử dụng trong dự án, chưa phải kết quả chứng minh cải thiện chất lượng. Ưu tiên DeepFashion và Fashionpedia cho thử nghiệm thuộc tính; DeepFashion2 phù hợp bổ sung cho retrieval và khác biệt cách chụp.

## 1. DeepFashion — Category and Attribute Prediction

Trang tác giả công bố 289.222 ảnh, 50 loại trang phục và 1.000 thuộc tính, kèm bounding box.

- [Google Drive do trang benchmark liên kết](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg&usp=sharing).
- Chọn đúng benchmark **Category and Attribute Prediction**.
- Tải ảnh, Category Annotations, Attribute Annotations, Bounding Box Annotations và Train/Val/Test Partitions. Nếu có nhiều bản nhãn, giữ nguyên tên gốc để chọn đúng bản khi viết loader.
- Chưa cần High-Res Images, Landmark Annotations hoặc các benchmark khác cho thử nghiệm đầu tiên.
- Lưu file tải tại `Data/downloads/deepfashion/`.

Nhãn thuộc tính không tự tạo thành bộ nhãn phong cách casual/formal/sporty. Cần kiểm tra từ vựng, độ tin cậy và mapping trước khi huấn luyện.

## 2. Fashionpedia

Trang dự án công bố 48.825 ảnh, 27 nhóm trang phục chính, 19 bộ phận trang phục và 294 thuộc tính chi tiết. Annotation gồm mask và thuộc tính gắn với từng đối tượng.

Tải các file sau từ [kho phân phối chính thức của CVDF](https://github.com/cvdfoundation/fashionpedia#download):

- [train2020.zip](https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip)
- [val_test2020.zip](https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip)
- [instances_attributes_train2020.json](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json)
- [instances_attributes_val2020.json](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json)

Lưu tại `Data/downloads/fashionpedia/`. Gói `val_test2020.zip` chứa cả ảnh validation và test; khi đánh giá cục bộ chỉ dùng những ảnh có trong annotation validation. Annotation theo đối tượng ở trên đủ cho hướng crop/mask từng món đồ; chưa cần tải thêm bản global attributes.

## 3. DeepFashion2

Kho chính thức mô tả khoảng 491 nghìn ảnh thuộc 13 loại trang phục, có ảnh shop và người dùng, bounding box, mask và thông tin ghép cặp.

- [Google Drive do tác giả cung cấp](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing).
- Tải gói **train** và **validation**, gồm ảnh và annotations. Chưa cần test cho thử nghiệm đầu tiên.
- Theo hướng dẫn tác giả, cần [điền form để nhận mật khẩu giải nén](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform?usp=sf_link).
- Lưu tại `Data/downloads/deepfashion2/`.

**Ý nghĩa nhãn `style`:** mã biến thể màu/in/logo trong các ảnh có cùng `pair_id`, không phải nhãn phong cách toàn cục. Chỉ ghép positive pair khi cùng `pair_id` và cùng `style > 0`, theo [mô tả của tác giả](https://github.com/switchablenorms/DeepFashion2#data-organization). Không gộp mọi đối tượng có `style = 1` thành một phong cách.

## Quy ước tiếp nhận

Các thư mục `Data/downloads/deepfashion/`, `Data/downloads/fashionpedia/` và `Data/downloads/deepfashion2/` đã được tạo để nhận file. Giữ tên file gốc; không ghi đè dữ liệu VISUELLE/Myntra. Khi có dữ liệu, giải nén vào thư mục riêng tương ứng dưới `Data/extracted/` và giữ cấu trúc do tác giả cung cấp.

Trước khi train:

1. Kiểm tra ảnh hỏng, cấu trúc nhãn, phân bố lớp và ảnh trùng giữa các nguồn.
2. Tôn trọng partition chính thức; với split mới, nhóm theo product identity/pair để cùng món đồ không xuất hiện ở cả train và validation/test.
3. Chuẩn hóa category/attribute có mapping rõ ràng; nhãn thiếu hoặc không tương đương không được coi là negative.
4. Crop/mask trang phục nhất quán; lấy mẫu cân bằng theo nguồn để hạn chế học nền/cách chụp.
5. So sánh mô hình frozen hiện tại với bản fine-tuned bằng retrieval, nhận diện thuộc tính và chất lượng/độ ổn định cụm trên tập giữ riêng. Chọn cấu hình trên validation, giữ tập test riêng đến lần đánh giá cuối.
6. Nếu đổi encoder hoặc centroid, xây lại style panel và train lại dự báo trong một thí nghiệm mới. Giữ kết quả nghiên cứu đã báo cáo để đối chiếu.

Ba nguồn trên bổ sung tín hiệu thị giác; chưa được dùng làm chuỗi doanh số cho dự báo 4 tuần. Việc tải về cũng chưa tích hợp dữ liệu vào ứng dụng. Điều kiện sử dụng theo trang của từng tác giả.
