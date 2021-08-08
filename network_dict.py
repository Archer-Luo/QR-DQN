network_dictionary = {

    'Nmodel': {
        'A': [[-1, 0], [0, -1]],
        'D': [[1, 1]],
        'alpha': [1.3 * 0.95, 0.4 * 0.95],
        'mu': [1, 1. / 2, 1],
        'name': 'Nmodel',
        'holding_cost': [3., 1.]
    },



    'reentrant': {
        'A': [[-1, 1, 0], [0, -1, 1], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [1.0, 0.0, 0.0],
        'mu': [5.0, 1 / 0.9, 1 / 0.75],
        'name': 'reentrant'
    },

    'criss_cross': {
        'A': [[-1, 1, 0], [0, -1, 0], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [0.9, 0.0, 0.9],
        'mu': [2.0, 1.0, 2.0],
        'name': 'criss_cross'
    },
    'criss_crossIH': {
        'A': [[-1, 1, 0], [0, -1, 0], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [0.9, 0.0, 0.9],
        'mu': [2.0, 1.5, 2.0],
        'name': 'criss_crossIH'
    },
    'criss_crossBM': {
        'A': [[-1, 1, 0], [0, -1, 0], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [0.6, 0.0, 0.6],
        'mu': [2.0, 1.0, 2.0],
        'name': 'criss_crossBM'
    },
    'criss_crossIM': {
        'A': [[-1, 1, 0], [0, -1, 0], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [0.6, 0.0, 0.6],
        'mu': [2.0, 1.5, 2.0],
        'name': 'criss_crossIM'
    },

    'criss_crossBL': {
        'A': [[-1, 1, 0], [0, -1, 0], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [0.3, 0.0, 0.3],
        'mu': [2.0, 1., 2.0],
        'name': 'criss_crossBL'
    },

    'criss_crossIL': {
        'A': [[-1, 1, 0], [0, -1, 0], [0, 0, -1]],
        'D': [[1, 0, 1], [0, 1, 0]],
        'alpha': [0.3, 0.0, 0.3],
        'mu': [2.0, 1.5, 2.0],
        'name': 'criss_crossIL'
    },

    '6-classes_re': {
        'A': [[-1, 0, 0, 1, 0, 0],
              [0, -1, 0, 0, 1, 0],
              [0, 0, -1, 0, 0, 1],
              [0, 1, 0, -1, 0, 0],
              [0, 0, 1, 0, -1, 0],
              [0, 0, 0, 0, 0, -1]],

        'D': [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        'alpha': [9. / 140., 0, 0, 0, 0, 0],
        'mu': [1. / 8, 1. / 2., 1. / 4, 1. / 6, 1. / 7, 1.],
        'name': '6-classes_re'

    },



    '6-classes': {
        'A': [[-1, 0, 0, 1, 0, 0],
              [0, -1, 0, 0, 1, 0],
              [0, 0, -1, 0, 0, 1],
              [0, 1, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, -1]],

        'D': [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        'alpha': [9. / 140., 0, 9. / 140., 0, 0, 0],
        'mu':[1. / 8, 1./2., 1. / 4, 1. / 6, 1. / 7, 1.],
        'name': '6-classes'

    },

    '9-classes_re':{
        'A':  [[-1, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, -1, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, -1, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, -1, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, -1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, -1, 0, 0, 1],
               [0, 1, 0, 0, 0, 0, -1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, -1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, -1]],

         'D':  [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 1]],
         'alpha': [9. / 140., 0, 0, 0, 0, 0, 0, 0, 0],
         'mu': [1. / 8, 1./2., 1. / 4, 1. / 6, 1. / 7, 1.,1. / 8, 1./2., 1. / 4 ],
         'name': '9-classes_re'
    },

    '9-classes': {
        'A': [[-1, 0, 0, 1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 0, -1, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, -1]],

        'D': [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1]],
        'alpha': [9. / 140., 0, 9. / 140., 0, 0, 0, 0, 0, 0],
        'mu': [1. / 8, 1. / 2., 1. / 4, 1. / 6, 1. / 7, 1., 1. / 8, 1. / 2., 1. / 4],
        'name': '9-classes'
    },



    '12-classes':{
        'A':  [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]],

         'D':  [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1,  0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
         'alpha': [9. / 140., 0, 9. / 140., 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'mu': [1. / 8, 1./2., 1. / 4, 1. / 6, 1. / 7, 1.,1. / 8, 1./2., 1. / 4 , 1. / 6, 1. / 7, 1.],
         'name': '12-classes'
    },

    '12-classes_re': {
        'A': [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]],

        'D': [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
        'alpha': [9. / 140., 0, 0., 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'mu': [1. / 8, 1. / 2., 1. / 4, 1. / 6, 1. / 7, 1., 1. / 8, 1. / 2., 1. / 4, 1. / 6, 1. / 7, 1.],
        'name': '12-classes_re'
    },


    '15-classes':{
        'A':  [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]],

         'D':  [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1,  0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
         'alpha': [9. / 140., 0, 9. / 140., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'mu':[1. / 8, 1./2., 1. / 4, 1. / 6, 1. / 7, 1.,1. / 8, 1./2., 1. / 4 , 1. / 6, 1. / 7, 1., 1. / 8, 1./2., 1. / 4],
         'name': '15-classes'
    },

    '15-classes_re': {
        'A': [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]],

        'D': [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
        'alpha': [9. / 140., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'mu': [1. / 8, 1. / 2., 1. / 4, 1. / 6, 1. / 7, 1., 1. / 8, 1. / 2., 1. / 4, 1. / 6, 1. / 7, 1., 1. / 8,
               1. / 2., 1. / 4],
        'name': '15-classes_re'
    },





    '18-classes':{
        'A':  [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]],

         'D':  [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],

         'alpha': [9. / 140., 0, 9. / 140., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'mu':[1. / 8, 1./2., 1. / 4, 1. / 6, 1. / 7, 1.,1. / 8, 1./2., 1. / 4 , 1. / 6, 1. / 7, 1., 1. / 8, 1./2., 1. / 4, 1. / 6, 1. / 7, 1.],
         'name': '18-classes'
    },

    '21-classes': {
        'A': [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
              ],

        'D': [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]],

        'alpha': [9. / 140., 0, 9. / 140., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'mu': [1. / 8, 1. / 2., 1. / 4,
               1. / 6, 1. / 7, 1.,
               1. / 8, 1. / 2., 1. / 4,
               1. / 6, 1. / 7, 1.,
               1. / 8, 1. / 2., 1. / 4,
               1. / 6, 1. / 7, 1.,
               1. / 8, 1. / 2., 1. / 4],
        'name': '21-classes'
    },

}