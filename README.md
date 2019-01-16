# DMFinalProject

This is a project that uses OPTICS clustering algorithm to cluster footballer faces and diabetic patients' data.

## Requirements : 

Install scipy==0.21dev0 with : 

    pip install git+https://github.com/scikit-learn/scikit-learn.git 
    
unless OPTICS is now part of the stable releas

Install all the rest of the requirements with:

    pip install -r requirements.txt

## Getting Started

### File structure

**Datasets**

    *dataset_diabetes*

        - diabetic_data.csv : 130 US hospital data from 1999 to 2008[1](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

        - IDs_mapping.csv   : mapping for admission type, discharge disposition and admission source

    *footballers*   : 124 footballers' photo (Neymar Jr., Lionel Messi, Cristiano Ronaldo, Luis Suarez, and Mohamed Salah)
        *Predict*   : 5 footballers' photo to use for prediction (Neymar Jr., Lionel Messi, Cristiano Ronaldo, Luis Suarez, and Mohamed Salah)
        
**Ressources**

    - footballers_encodings.pickle          : Encodings of the footballers after using *encode_faces.py*

    - footballers_predict_encodings.pickle  : Encodings of the footballers for prediction after using *encode_faces.py*

    - GUFD_encodings.pickle                 : Encodings of the GUFD photo after using *encode_faces.py*

    - shape_predictor_68_face_landmarks.dat : Used to extract facial features in *encode_features.py*

- *encode_faces.py*           : used to encode the face in an image as a 128-d vector
- *encode_features.py*       : used to encode the facial features of a face as a 7-d vector
- *faces_clustering.py*       : used to cluster the footballers' faces
- *diabetic_clustering.py*    : used to cluster the diabetic's data
- *similarity_clustering.py*  : used to cluster the GUFD dataset based on facial features similiarities
- *optics.py*                 : contains the clustering and predicting methods

- *requirements.txt*          : file containing all the python packages requirements

## Authors

* **Victor MEUNIER** - *DMFinalProject* - [MrEliptik](https://github.com/MrEliptik)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details