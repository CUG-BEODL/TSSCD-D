# TSSCD-D

<div align="center">
  <h2><b>
  Monitoring of mangrove dynamic change with Landsat time series from 1987 to 2022 in the Beibu Gulf, China
  </b></h2>
</div>

<div align="center">

[[Paper]](to be added) · [[Code]](https://github.com/yourname/TSSCD-D)

</div>

The official PyTorch implementation of the paper  
**"Monitoring of mangrove dynamic change with Landsat time series from 1987 to 2022 in the Beibu Gulf, China"**,  
accepted by the *International Journal of Applied Earth Observation and Geoinformation*.

> **TSSCD-D** is an enhanced version of **TSSCD**, incorporating **dilated temporal convolutions** to enlarge the temporal receptive field, enabling robust long-term change detection from dense Landsat time series.

> TSSCD → **TSSCD-D** ✨  
> *Temporal Semantic Segmentation for Change Detection with Dilated Temporal Modeling*

---

## Update

- 2026.01.24:  Code released.

---

## Overview

TSSCD-D is designed for **long-term land-cover and ecosystem change detection** from dense remote sensing time series.  
The framework integrates:

- Temporal semantic segmentation for per-pixel trajectory modeling  
- Multi-scale dilated temporal convolutions for long-range dependency modeling  


## Framework

The proposed framework consists of a TSSCD backbone augmented with a **Multi-scale Dilated Temporal Convolution (MDTC)** module, enabling effective modeling of long-term temporal dependencies while maintaining computational efficiency.

<p align="center">
  <img src="img/model.jpg" width="700px">
</p>

---

## Usage

```python
python main.py
```
## Study Area & Data

- **Study area**: Beibu Gulf (China–Vietnam coast)  
- **Data source**: Landsat 4/5/7/8/9 surface reflectance  
- **Temporal coverage**: 1987–2022  
- **Temporal resolution**: Quarterly (144 time steps)

---

## Contact

If you have any questions or suggestions, feel free to contact us.

- **Email**: qinglinyang@cug.edu.cn

---



