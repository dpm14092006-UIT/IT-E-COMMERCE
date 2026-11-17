# Outputs

Thư mục này chứa toàn bộ kết quả sinh ra từ pipeline (`scripts/`). Tài liệu này mô tả các **checkpoint fine-tune** (`*.pt`) trong `embeddings/`.

## Checkpoint fine-tune (Myntra)

Hai encoder được fine-tune trên ảnh Myntra (Women + Girls, Apparel) với bài toán phân loại đa nhiệm **article type** (29 lớp) và **colour** (26 lớp). Sau đó backbone được dùng để trích xuất embedding cho 5.354 sản phẩm VISUELLE 2.0.

| | ResNet-50 fine-tuned | FashionCLIP fine-tuned |
|---|---|---|
| File checkpoint | `embeddings/resnet50ft_backbone.pt` | `embeddings/fashionclipft_backbone.pt` |
| Kích thước | 94.348.979 bytes (~90 MB) | 349.897.588 bytes (~334 MB) |
| SHA-256 | `6145650c7316a4dc6bec1b8e042f5e5e14fd5cf2f56f01688afa9bf09ec5f1ea` | `262c06a449718e062a93d5f7c52d1e5932745780fcbe61869c38bb2fc218282b` |
| Backbone gốc | ResNet-50, weights `IMAGENET1K_V2` | `patrickjohncyh/fashion-clip` (`vision_model`) |
| Layer được train | `layer4` | encoder layer cuối + `post_layernorm` (phần còn lại freeze) |
| Epochs | 3 | 3 |
| Learning rate | 1e-3 | 1e-5 |
| Batch size | 64 | 32 |
| Embedding dim | 2048 | 512 |
| Seed | 2026 | 2026 |

Dữ liệu fine-tune: `Data/images(SMALL).zip` (Myntra), 8.850 ảnh, `min_class_count = 30`, genders `Women` và `Girls`, master category `Apparel`.

### Lịch sử huấn luyện

**ResNet-50 fine-tuned**

| Epoch | train_loss | val_acc (type) | val_acc (colour) |
|---|---|---|---|
| 1 | 2.7165 | 0.7288 | 0.6192 |
| 2 | 1.6803 | 0.7458 | 0.6520 |
| 3 | 1.3076 | 0.7514 | 0.6497 |

**FashionCLIP fine-tuned**

| Epoch | train_loss | val_acc (type) | val_acc (colour) |
|---|---|---|---|
| 1 | 4.6627 | 0.6520 | 0.5864 |
| 2 | 2.8689 | 0.7514 | 0.6542 |
| 3 | 2.3700 | 0.7864 | 0.6746 |

### Kiểm tra embedding (G1)

| | ResNet-50 ft | FashionCLIP ft |
|---|---|---|
| Số sản phẩm (n) | 5.354 | 5.354 |
| Dim | 2048 | 512 |
| NaN / zero-norm | 0 / 0 | 0 / 0 |
| Norm min – max | 15.92 – 74.65 | 10.17 – 14.18 |

Chi tiết đầy đủ: `embeddings/*_training.json`, `embeddings/*_G1.json`, `manifests/02b_finetune_*.json`.

### Môi trường huấn luyện

Python 3.12.16 (WSL2), torch 2.15.0+cpu, torchvision 0.29.2+cpu, transformers 4.57.6. Config hash: `b30bce42858dd4634b1dbf36999ca62b3bd2e5f05521330fa29d5756f24aca9c`.

### Lưu ý: file `.pt` không nằm trên GitHub

- `fashionclipft_backbone.pt` (~334 MB) vượt giới hạn 100 MB mỗi file của GitHub.
- `resnet50ft_backbone.pt` (~90 MB) không được push để tránh làm repo phình to.

Cách tái tạo checkpoint (đối chiếu SHA-256 ở trên để xác nhận):

```bash
python scripts/02b_finetune_resnet.py
```

Các embedding đã trích xuất (`embeddings/*.npy`) và toàn bộ kết quả downstream vẫn có trong repo, nên không cần checkpoint để xem lại kết quả.
