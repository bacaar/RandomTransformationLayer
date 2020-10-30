# RandomTransformationLayer
This code represents a peprocessing layer for image augmentation in neuronal networks based on the NiftyNet framework. It has been developed as part of my bachelor thesis at the UMIT university (https://www.umit-tirol.at). It combines different image transformations (flip, rotate, scale) and elastic deformation, in order to reduce the amount of interpolations needed.

### Usage
The usage does not differ from any other preprocessing layer provided by NiftyNet.

### Cython
The transformations work mostly with methods from NumPy and SciPy, however, to use the elastic deformation, Cython (https://cython.org/) is needed. This Cython file (transform\_cy.pyx) has to be compiled before usage on every singe PC individually. To do so, run the provided setup.py file with python3 in your terminal:

```python
python3 setup.py build_ext --inplace
```

Then copy the created .so file in your project directory. Don't worry if the IDE reports it can't find the file, it will find it if you link the file correctly (the import path probably needs to be adjusted)
