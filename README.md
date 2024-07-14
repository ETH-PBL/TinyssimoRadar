# TinyssimoRadar

TinyssimoRadar proposes a low-power in-ear Hand-Gesture Recognition system based on mm-wave radars, efficient spatial and temporal Convolutional Neural Networks and an energy-optimized hardware design. 🤖📡✋
The hardware implementation is based on our miniaturized wearable platform [VitalCore](https://github.com/ETH-PBL/VitalCore).

The dataset used in this work is available for download from ETH Library at [this link](https://www.research-collection.ethz.ch/handle/20.500.11850/672242). More information on the data format and relative licence can be found in the link above.

## Citation

If you find this work, the hardware and/or the dataset useful for your research, please consider citing our paper:

[Andrea Ronco, Philipp Schilk, Michele Magno, "TinyssimoRadar: In-Ear Hand Gesture Recognition with Ultra-Low Power mmWave Radars", *IoTDI24*, Year.](https://ieeexplore.ieee.org/abstract/document/10562162) 📚

```bibtex
@inproceedings{TinyssimoRadar,
  author={Ronco, Andrea and Schilk, Philipp and Magno, Michele},
  booktitle={2024 IEEE/ACM Ninth International Conference on Internet-of-Things Design and Implementation (IoTDI)}, 
  title={TinyssimoRadar: In-Ear Hand Gesture Recognition with Ultra-Low Power mmWave Radars}, 
  year={2024},
  volume={},
  number={},
  pages={192-202},
  doi={10.1109/IoTDI61053.2024.00021}
}
```

## Getting Started

To start using TinyssimoRadar:

1. Set up a Python environment with pip installed.
2. Create a virtual environment using `virtualenv` or `venv`:
   ```
   virtualenv venv
   ```
   or
   ```
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run TinyssimoRadar
   ```
   pyhon -m main.py train quantize evaluate
   ```

   Several model parameters are availabe, get more information via --help
   ```
   python main.py --help
   ```

## License

This project is licensed under the Published under GNU GPLv3 license. - see the [LICENSE](LICENSE) file for details. 📝
