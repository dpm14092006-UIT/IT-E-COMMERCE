# 3. Methodology

## 3.1. Visual Feature Extraction and Style Clustering Using ResNet-50

### 3.1.1. ResNet-50 Architecture and Feature Extraction

Nghiên cứu sử dụng mạng nơ-ron tích chập sâu ResNet-50 để chuyển đổi hình ảnh sản phẩm thời trang thành các vector đặc trưng phục vụ phân tích mức độ tương đồng thị giác. ResNet-50 được phát triển dựa trên cơ chế học phần dư (Residual Learning), trong đó các kết nối tắt cho phép thông tin đầu vào được truyền trực tiếp qua những khối tích chập, hỗ trợ quá trình lan truyền gradient và tối ưu mạng sâu (He et al., 2016). Nghiên cứu sử dụng ResNet-50 của TorchVision (biến thể ResNet V1.5), khởi tạo bằng bộ trọng số ImageNet-1K IMAGENET1K_V2. Kiến trúc gồm 16 khối residual bottleneck được tổ chức thành bốn giai đoạn theo cấu hình 3, 4, 6 và 3 khối, với số kênh đầu ra lần lượt là 256, 512, 1.024 và 2.048. Mỗi khối bottleneck sử dụng ba phép tích chập có kích thước 1 × 1, 3 × 3 và 1 × 1 để biến đổi số kênh và trích xuất thông tin không gian. Đầu ra của nhánh tích chập được cộng với đầu ra của nhánh kết nối tắt trước khi áp dụng ReLU.

$$
 \mathbf{y}=\operatorname{ReLU}\left(F(\mathbf{x})+S(\mathbf{x})\right) 
$$

Trong đó, $F(\mathbf{x})$ là phép biến đổi đặc trưng qua các lớp tích chập, $S(\mathbf{x})$ là phép ánh xạ qua kết nối tắt và $\mathbf{y}$ là đầu ra sau hàm kích hoạt ReLU. Cấu trúc tổng thể và nguyên lý hoạt động của khối bottleneck được trình bày trong Figure 1.
Trước khi trích xuất đặc trưng, hình ảnh được chuyển về không gian màu RGB. Ảnh có vùng trong suốt được ghép lên nền trắng trước khi chuyển đổi. Ảnh được thay đổi kích thước sao cho cạnh ngắn bằng 232 pixel, sau đó lấy vùng trung tâm 224 × 224 pixel và chuẩn hóa theo thống kê ImageNet. Tầng phân loại cuối cùng của ResNet-50 được loại bỏ nhằm thu nhận biểu diễn trực tiếp sau Global Average Pooling. Từ 2.048 bản đồ đặc trưng có kích thước 7 × 7, phép lấy trung bình toàn cục tạo ra một vector 2.048 chiều cho mỗi ảnh theo công thức sau.

$$
 z_{i,c}=\frac{1}{49}\sum_{h=1}^{7}\sum_{w=1}^{7}A_{i,c,h,w} 
$$

Trong đó, $A_{i,c,h,w}$ là giá trị kích hoạt tại vị trí $(h,w)$ thuộc kênh đặc trưng thứ $c$ của ảnh thứ $i$. Các vector được kiểm tra về số chiều và tính hợp lệ, sau đó chuẩn hóa L2 theo biểu thức $\widehat{\mathbf{z}}_i=\mathbf{z}_i/\|\mathbf{z}_i\|_2$, trong đó $\|\mathbf{z}_i\|_2$ là chuẩn Euclid của vector. Phép chuẩn hóa đưa các biểu diễn hợp lệ về độ dài đơn vị, hạn chế ảnh hưởng của sự khác biệt về độ lớn khi tính khoảng cách. Quá trình trích xuất sử dụng trọng số cố định. Với cấu hình pretrained, tính nhất quán được kiểm tra bằng độ tương đồng cosine giữa các vector thu được khi trích xuất lặp lại trên cùng nhóm ảnh.

### 3.1.2. Domain-Specific Fine-Tuning of ResNet-50

Để khảo sát ảnh hưởng của việc thích nghi mô hình với lĩnh vực thời trang, nghiên cứu xây dựng thêm một cấu hình ResNet-50 được tinh chỉnh bằng phương pháp học chuyển giao có giám sát. Chiến lược này dựa trên đặc điểm phân cấp của CNN, trong đó những tầng đầu thường học các đặc trưng thị giác tổng quát, còn những tầng sâu có khả năng hình thành biểu diễn chuyên biệt hơn đối với nhiệm vụ huấn luyện (Yosinski et al., 2014). Tập Myntra được lọc theo nhóm Women và Girls trong danh mục Apparel. Nghiên cứu lần lượt loại các lớp có dưới 30 ảnh theo loại trang phục rồi theo màu cơ bản. Sau lọc, 8.850 ảnh thuộc 29 loại trang phục và 26 nhóm màu được chia ngẫu nhiên theo tỷ lệ 90:10 bằng seed 2026, với 7.965 ảnh huấn luyện và 885 ảnh xác thực. Trong quá trình tinh chỉnh, chỉ các tham số của giai đoạn residual cuối (layer4) và hai đầu phân loại tuyến tính được cập nhật. Các lớp BatchNorm thuộc phần backbone đóng băng được đặt ở chế độ đánh giá để giữ cố định thống kê đang chạy. Hai bộ phân loại thực hiện đồng thời nhiệm vụ nhận diện loại sản phẩm với 29 lớp và màu sắc cơ bản với 26 lớp, sử dụng chung biểu diễn 2.048 chiều. Cách tiếp cận học đa nhiệm cho phép tín hiệu từ những nhiệm vụ liên quan cùng tham gia điều chỉnh mạng trích xuất đặc trưng (Caruana, 1997).
Trong giai đoạn huấn luyện và xác thực hai đầu phân loại, ảnh được chuyển sang RGB và resize cạnh ngắn về 256 pixel. Ảnh huấn luyện được cắt ngẫu nhiên với tỷ lệ diện tích từ 70% đến 100%, đưa về 224 × 224 pixel và lật ngang ngẫu nhiên. Ảnh xác thực được cắt vùng trung tâm 224 × 224 pixel. Cả hai tập sử dụng thống kê chuẩn hóa ImageNet. Hàm mục tiêu được xây dựng bằng tổng hai thành phần Cross-Entropy với trọng số bằng nhau.

$$
 \mathcal{L}_{total}=\mathcal{L}_{type}+\mathcal{L}_{colour} 
$$

Trong đó, $\mathcal{L}_{type}$ và $\mathcal{L}_{colour}$ lần lượt là mất mát của nhiệm vụ phân loại loại sản phẩm và màu sắc. Quá trình tinh chỉnh sử dụng AdamW với tốc độ học 0,001, hệ số suy giảm trọng số 0,0001, kích thước lô 64 ảnh và ba chu kỳ huấn luyện (Loshchilov & Hutter, 2019). Quy trình sử dụng trạng thái backbone ở cuối lịch huấn luyện ba epoch để trích xuất đặc trưng. Hai đầu phân loại được bỏ qua ở bước này. Cả hai cấu hình pretrained và fine-tuned đều tạo ra vector 2.048 chiều thông qua cùng quy trình Global Average Pooling và chuẩn hóa L2, bảo đảm sự nhất quán về kiến trúc đầu ra khi so sánh chất lượng biểu diễn.

### 3.1.3. Visual Style Clustering Using K-Means

Các vector đặc trưng được trích xuất từ ResNet-50 được phân nhóm bằng thuật toán K-Means nhằm xác định những sản phẩm có biểu diễn thị giác tương đồng. K-Means thực hiện phân hoạch không giám sát thông qua việc tối thiểu hóa tổng bình phương khoảng cách Euclid giữa các quan sát và tâm cụm gần nhất (Lloyd, 1982). Với $n$ sản phẩm và $K$ cụm, hàm mục tiêu là tổng bình phương khoảng cách đến tâm gần nhất.

$$
 \min_{\boldsymbol{\mu}_1,\ldots,\boldsymbol{\mu}_K} \sum_{i=1}^{n} \min_{1\leq k\leq K} \left\|\widehat{\mathbf{z}}_i-\boldsymbol{\mu}_k\right\|_2^2 
$$

Trong đó, $\widehat{\mathbf{z}}_i$ là vector đặc trưng đã chuẩn hóa của sản phẩm thứ $i$, còn $\boldsymbol{\mu}_k$ là tâm cụm thứ $k$. Nghiên cứu sử dụng K-Means của scikit-learn để khảo sát 10, 15, 20, 25 và 30 cụm. Với mỗi mô hình và mỗi giá trị K, mỗi lần fit sử dụng 20 lần khởi tạo và tối đa 500 vòng lặp cho mỗi lần khởi tạo. Trong một lần fit, nghiệm có inertia thấp nhất được giữ lại. Nghiệm tại seed 2026 được dùng để báo cáo Silhouette Score, Davies–Bouldin Index, kích thước cụm, NMI và số mốc dự báo đủ điều kiện, đồng thời cung cấp tâm cụm để gán nhóm sản phẩm. Đối với VISUELLE 2.0, các tâm cụm chỉ được ước lượng từ những sản phẩm thuộc tập phát triển, xác định theo ngày mở bán đầu tiên không muộn hơn ngày 31/12/2018. Sau khi xây dựng, các tâm cụm được giữ cố định để gán nhóm cho những sản phẩm xuất hiện sau thời điểm này, nhằm hạn chế việc sử dụng thông tin từ tương lai. Một cấu hình đủ điều kiện khi Mean ARI ≥ 0,70, mỗi cụm có ít nhất 30 sản phẩm trong tập phát triển và mỗi cụm có ít nhất 20 mốc dự báo đủ điều kiện trong giai đoạn phát triển, tính đến tuần bắt đầu ngày 31/12/2018. Tại một mốc dự báo, cụm phải có sản phẩm đang hoạt động, đã trải qua ít nhất 16 tuần tính từ tuần hoạt động đầu tiên (kể cả tuần hiện tại) và có chỉ số doanh số trên mỗi sản phẩm hoạt động tính được trong cửa sổ bốn tuần kết thúc tại mốc đó. Cửa sổ này yêu cầu cả bốn tuần đều có sản phẩm hoạt động. Các cấu hình đạt ngưỡng được xem xét bằng ảnh đại diện để chọn mức độ chi tiết phù hợp cho phân nhóm thị giác. Bảng ảnh của mỗi cụm gồm tối đa 12 ảnh gần tâm nhất và tối đa 12 ảnh chọn ngẫu nhiên từ phần còn lại.

### 3.1.4. Clustering Evaluation and Validation

Chất lượng phân cụm được đánh giá thông qua bốn chỉ số bổ sung cho nhau. Silhouette Score đo mức độ kết dính nội cụm và phân tách giữa các cụm (Rousseeuw, 1987), trong khi Davies–Bouldin Index đánh giá mức phân tán nội cụm tương quan với khoảng cách giữa những tâm cụm (Davies & Bouldin, 1979). Độ ổn định được báo cáo bằng trung bình và độ lệch chuẩn của Adjusted Rand Index (ARI) trên 45 cặp phân hoạch từ 10 lần fit với seed từ 2026 đến 2035 (Hubert & Arabie, 1985). Ngoài ra, Normalized Mutual Information (NMI) được sử dụng để đánh giá mức tương ứng giữa các nhóm được hình thành và những thuộc tính sản phẩm đã biết (Vinh et al., 2010). Trên VISUELLE 2.0, Silhouette Score, Davies–Bouldin Index, ARI và NMI được tính trên 3.549 sản phẩm thuộc tập phát triển. NMI được tính với danh mục, màu sắc và chất liệu trên VISUELLE, và với loại sản phẩm và màu sắc cơ bản trên Myntra.
Để đối chiếu hai cấu hình ResNet-50, nghiên cứu sử dụng cùng 885 ảnh Myntra thuộc tập xác thực. Với mỗi cấu hình, các vector đặc trưng được trích xuất và chuẩn hóa L2, sau đó một mô hình K-Means độc lập với 20 cụm được xây dựng trên chính tập biểu diễn tương ứng. NMI được tính bằng cách đối chiếu phân hoạch K-Means với nhãn thuộc tính của sản phẩm. Giao thức thực nghiệm được minh họa trong Figure 2. Do K-Means được xây dựng và đánh giá trên cùng tập ảnh Myntra, các chỉ số thu được mô tả cấu trúc cụm, độ ổn định và mức tương ứng với nhãn thuộc tính trong mẫu. Các tâm cụm Myntra được ước lượng độc lập với các tâm cụm VISUELLE 2.0.

# 4. Results

## 4.1. ResNet-50 Fine-Tuning Performance

Table 1 trình bày training loss trên 7.965 ảnh huấn luyện và độ chính xác trên 885 ảnh xác thực Myntra qua ba epoch fine-tuning.

**Table 1. Training loss and validation accuracy of fine-tuned ResNet-50.**

| Epoch | Training loss | Garment type accuracy (%) | Colour accuracy (%) |
| --- | --- | --- | --- |
| 1 | 2,717 | 72,88 | 61,92 |
| 2 | 1,680 | 74,58 | 65,20 |
| 3 | 1,308 | 75,14 | 64,97 |

Training loss giảm 51,9% từ epoch 1 đến epoch 3, trong khi độ chính xác phân loại trang phục tăng 2,26 điểm phần trăm. Độ chính xác nhận diện màu sắc đạt cao nhất 65,20% tại epoch thứ hai, sau đó giảm nhẹ xuống 64,97%. Checkpoint sau epoch thứ ba được sử dụng để trích xuất đặc trưng cho 5.354 sản phẩm VISUELLE có ảnh hợp lệ. Toàn bộ vector đặc trưng có 2.048 chiều, không ghi nhận giá trị NaN hoặc vector có chuẩn bằng không.

## 4.2. Selection of the Number of Clusters

Phân cụm K-Means được đánh giá trên 3.549 sản phẩm thuộc tập phát triển VISUELLE với năm cấu hình $K \in \{10,15,20,25,30\}$. Độ ổn định được đo bằng Mean Adjusted Rand Index (ARI) giữa các phân hoạch thu được từ 10 seed. Kết quả so sánh hai phiên bản ResNet-50 được trình bày trong Table 2.

**Table 2. Clustering stability and eligibility across different values of K on VISUELLE.**

| K | Pretrained Mean ARI | Eligible | Fine-tuned Mean ARI | Eligible |
| --- | --- | --- | --- | --- |
| 10 | 0,869 | Yes | 0,938 | Yes |
| 15 | 0,699 | No | 0,840 | Yes |
| 20 | 0,723 | Yes | 0,684 | No |
| 25 | 0,669 | No | 0,665 | No |
| 30 | 0,601 | No | 0,617 | No |

Note. Một cấu hình được chấp nhận khi Mean ARI ≥ 0,70, kích thước cụm nhỏ nhất ≥ 30 sản phẩm và mỗi cụm có ít nhất 20 mốc dự báo đủ điều kiện.

Kết quả cho thấy phiên bản pretrained đáp ứng toàn bộ tiêu chí tại $K=10$ và $K=20$, trong khi phiên bản fine-tuned đáp ứng tại $K=10$ và $K=15$. Cả hai cấu hình đạt Mean ARI cao nhất tại $K=10$. Khi xem ảnh đại diện của cấu hình pretrained, phương án 10 cụm cho mức phân nhóm quá thô để diễn giải các nhóm sản phẩm. Phương án 20 cụm được chọn trước bước dự báo vì cho mức phân nhóm chi tiết hơn theo đánh giá trực quan và vẫn đáp ứng cả ba ngưỡng. Cấu hình này được giữ làm cấu hình chính sau khi bổ sung thí nghiệm fine-tuning trên Myntra.

Ở cấu hình $K=20$, kích thước các cụm nằm trong khoảng 111–286 sản phẩm đối với pretrained và 106–299 sản phẩm đối với fine-tuned. Số mốc dự báo đủ điều kiện thấp nhất lần lượt là 70 và 77, đều vượt ngưỡng yêu cầu. Tuy nhiên, Mean ARI của fine-tuned chỉ đạt 0,684, thấp hơn ngưỡng 0,70. Vì vậy, trong hai cấu hình ResNet-50 tại 20 cụm, bản pretrained là cấu hình duy nhất đáp ứng đồng thời cả ba tiêu chí lựa chọn. Các tâm cụm của cấu hình này sau đó được sử dụng để gán nhóm cho toàn bộ 5.354 sản phẩm VISUELLE hợp lệ.

## 4.3. Clustering Quality and Attribute Agreement

Chất lượng phân cụm tại $K=20$ được đánh giá thông qua Silhouette Score, Davies–Bouldin Index và Mean ARI. Mức tương ứng giữa các cụm và thuộc tính sản phẩm được đo bằng Normalized Mutual Information (NMI). Table 3 trình bày kết quả trên 3.549 sản phẩm thuộc tập phát triển VISUELLE và 885 ảnh thuộc tập xác thực Myntra.

**Table 3. Clustering quality and attribute agreement of ResNet-50 at K = 20.**

| Evaluation metric | VISUELLE Pretrained | VISUELLE Fine-tuned | Myntra Pretrained | Myntra Fine-tuned |
| --- | --- | --- | --- | --- |
| Silhouette Score ↑ | 0,066 | 0,085 | 0,080 | 0,097 |
| Davies–Bouldin Index ↓ | 2,826 | 2,531 | 2,609 | 2,307 |
| Mean ARI ↑ | 0,723 | 0,684 | 0,544 | 0,597 |
| ARI standard deviation | 0,050 | 0,048 | 0,052 | 0,052 |
| NMI – Category | 0,349 | 0,249 | 0,475 | 0,474 |
| NMI – Colour | 0,099 | 0,232 | 0,173 | 0,384 |
| NMI – Fabric | 0,330 | 0,241 | — | — |

Trên VISUELLE, fine-tuning làm tăng Silhouette Score từ 0,066 lên 0,085 và giảm Davies–Bouldin Index từ 2,826 xuống 2,531. Silhouette Score của cả hai cấu hình đều gần 0, cho thấy mức tách biệt giữa các cụm còn yếu theo chỉ số này. Mean ARI giảm từ 0,723 xuống 0,684. Đồng thời, NMI màu sắc tăng từ 0,099 lên 0,232, trong khi NMI danh mục giảm từ 0,349 xuống 0,249 và NMI chất liệu giảm từ 0,330 xuống 0,241.

Trên 885 ảnh Myntra xác thực, fine-tuning cải thiện cả Silhouette Score, Davies–Bouldin Index và Mean ARI. Đáng chú ý, NMI màu sắc tăng từ 0,173 lên 0,384, còn NMI loại trang phục gần như không thay đổi (0,475 so với 0,474). Kết quả trên hai tập ảnh đều ghi nhận sự gia tăng mức tương ứng giữa cụm và thuộc tính màu sắc sau fine-tuning. Mean ARI tăng trên Myntra và giảm trên VISUELLE.

Các chỉ số Myntra mô tả kết quả trong mẫu. Hai mô hình K-Means được ước lượng và đánh giá trên cùng 885 ảnh xác thực, vốn được giữ riêng khỏi quá trình cập nhật trọng số CNN.

## 4.4. Qualitative Assessment of Visual Style Clusters

Đánh giá trực quan các cụm được gán cho danh mục VISUELLE có ảnh hợp lệ bằng ResNet-50 pretrained tại $K=20$ cho thấy nhiều nhóm tập trung vào những đặc điểm thị giác có thể nhận diện. Chẳng hạn, cụm 12 chủ yếu gồm áo khoác (77%), cụm 2 tập trung các sản phẩm màu đen (76%), trong khi cụm 19 có tỷ lệ sản phẩm mang nhãn màu xanh dương (blue) đạt 96%. Các ví dụ này minh họa sự tập trung theo danh mục hoặc màu sắc ở từng cụm.

Cụm 7 đa dạng hơn về danh mục. Các nhóm phổ biến gồm áo dài tay (18%), váy dài (15%) và chân váy ngắn (12%). Tên cụm là mô tả dựa trên ảnh đại diện và hồ sơ thuộc tính của sản phẩm trong cụm.
