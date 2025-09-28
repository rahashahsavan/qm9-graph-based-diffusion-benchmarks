# GraphARM QM9 Training and Generation Guide

## مشکل حل شده: TypeError QM9.__init__()

مشکل اصلی این بود که کلاس `QM9` از `torch_geometric` پارامتر `remove_h` را پشتیبانی نمی‌کند. این مشکل با پردازش دستی داده‌ها حل شده است.

## نصب وابستگی‌ها

```bash
cd GraphARM
pip install -r requirements_qm9.txt
```

## مراحل استفاده

### ۱. تست بارگذاری داده‌ها

```bash
python test_qm9_loading.py
```

این اسکریپت:
- QM9 dataset را بارگذاری می‌کند
- هیدروژن‌ها را حذف می‌کند
- آمار dataset را نمایش می‌دهد

### ۲. آموزش مدل

```bash
python train_qm9.py
```

**تنظیمات:**
- `REMOVE_HYDROGEN = True`: حذف هیدروژن (پیش‌فرض)
- `REMOVE_HYDROGEN = False`: نگه‌داشتن هیدروژن

### ۳. تولید مولکول‌ها

```bash
python generate_molecules.py
```

## پیکربندی‌های موجود

### بدون هیدروژن (پیش‌فرض)
```python
REMOVE_HYDROGEN = True
# Max nodes: 9
# Atom types: C, N, O, F
# پیچیدگی: متوسط
```

### با هیدروژن
```python
REMOVE_HYDROGEN = False
# Max nodes: 29
# Atom types: H, C, N, O, F
# پیچیدگی: بالا
```

## خروجی‌ها

- **مدل آموزش‌دیده:** `qm9_denoising_network.pt`, `qm9_diffusion_ordering_network.pt`
- **مولکول‌های تولیدی:** `qm9_generated_10000.smi`
- **آمار مولکول‌ها:** وزن مولکولی، تعداد اتم‌ها، حلقه‌ها

## عیب‌یابی

### خطای "QM9.__init__() got an unexpected keyword argument 'remove_h'"
✅ **حل شده** - از پردازش دستی داده‌ها استفاده می‌شود

### خطای "Import wandb could not be resolved"
⚠️ **Warning** - نصب کنید: `pip install wandb`

### خطای حافظه
💡 **راه‌حل:** `batch_size` را کاهش دهید

## مثال استفاده

```python
# در train_qm9.py یا generate_molecules.py
REMOVE_HYDROGEN = True   # برای حذف هیدروژن
REMOVE_HYDROGEN = False  # برای نگه‌داشتن هیدروژن
```

## نکات مهم

1. **استاندارد صنعت:** اکثر کارهای QM9 هیدروژن را حذف می‌کنند
2. **کارایی:** بدون هیدروژن سریع‌تر و دقیق‌تر است
3. **دقت:** با هیدروژن کامل‌تر اما پیچیده‌تر است

## پشتیبانی

اگر مشکلی داشتید، ابتدا `test_qm9_loading.py` را اجرا کنید تا مطمئن شوید داده‌ها درست بارگذاری می‌شوند.
