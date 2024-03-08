# TinyssimoRadar

TinyssimoRadar proposes a low-power in-ear Hand-Gesture Recognition system based on mm-wave radars, efficient spatial and temporal Convolutional Neural Networks and an energy-optimized hardware design. 🤖📡✋
The hardware implementation is based on our miniaturized wearable platform [VitalCore](https://github.com/ETH-PBL/VitalCore).

## Citation

If you find this work useful, please consider citing our paper:

[Andrea Ronco, Philipp Schilk, Michele Magno, "TinyssimoRadar: In-Ear Hand Gesture Recognition with Ultra-Low Power mmWave Radars", *IoTDI24*, Year.](link_to_paper) 📚

## Implementation Details

TinyssimoRadar is based on a Temporal Convolutional Network (TCN) model. This model architecture is specifically tailored for time-series data and offers advantages in capturing temporal dependencies efficiently. The project targets low-power constrained devices, thus employs techniques such as quantization to optimize model performance while maintaining low power consumption.

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
