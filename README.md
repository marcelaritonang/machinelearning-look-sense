# Machine Learning Look Sense: Fashion Classification

**Machine Learning Look Sense** adalah proyek pembelajaran mesin yang dirancang untuk mengklasifikasikan item fashion dari citra. Proyek ini mencakup pipeline konversi model yang kompleks dari PyTorch ke ONNX, TensorFlow, hingga TensorFlow.js untuk memungkinkan deployment berbasis web. Anda dapat mencoba aplikasi web di: [looksense.vercel.app](https://looksense.vercel.app).

## ✨ Fitur

- **Klasifikasi Fashion**: Model mampu mengenali dan mengelompokkan item fashion ke dalam 7 kategori:
  - **Bags** (Tas)
  - **Bottomwear** (Pakaian Bawah)
  - **Dress** (Gaun)
  - **Headwear** (Aksesori Kepala)
  - **Shoes** (Sepatu)
  - **Topwear** (Pakaian Atas)
  - **Watches** (Jam Tangan)
- **Pipeline Model**:
  - **PyTorch**: Model dikembangkan dan dilatih menggunakan PyTorch.
  - **ONNX**: Model dikonversi ke format ONNX untuk interoperabilitas.
  - **TensorFlow**: ONNX dikonversi ke TensorFlow untuk mendukung framework ML populer.
  - **TensorFlow.js**: Model dioptimalkan untuk digunakan pada aplikasi berbasis web.
- **Web Deployment**: Model di-hosting di Vercel untuk aksesibilitas global.

---

## 🚀 Teknologi yang Digunakan

- **PyTorch**: Framework utama untuk melatih model deep learning.
- **ONNX**: Format model terbuka untuk interoperabilitas antar framework.
- **TensorFlow**: Framework untuk konversi model dan optimasi.
- **TensorFlow.js**: Untuk menjalankan model di browser.
- **Next.js & Vercel**: Teknologi frontend dan hosting untuk deployment aplikasi.

---

## 📂 Struktur Direktori

Berikut adalah struktur direktori utama proyek ini:

- **`data/`**: Dataset mentah untuk pelatihan dan pengujian model.
- **`models/`**: Model yang telah dilatih, termasuk file PyTorch, ONNX, dan TensorFlow.
- **`src/`**: Kode sumber utama untuk pelatihan, evaluasi, dan konversi model.
- **`notebooks/`**: Jupyter Notebooks untuk eksplorasi data, pengembangan model, dan analisis.
- **`web/`**: Kode frontend aplikasi berbasis Next.js untuk deployment.

---

## 📖 Pipeline Model

1. **Pelatihan di PyTorch**:
   Model dilatih menggunakan PyTorch untuk memanfaatkan fleksibilitas framework.

2. **Konversi ke ONNX**:
   Model PyTorch dikonversi ke format ONNX untuk interoperabilitas:
   ```bash
   torch.onnx.export(model, dummy_input, "model.onnx", ...)
