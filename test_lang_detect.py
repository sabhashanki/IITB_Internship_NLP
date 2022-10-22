import unittest
import lang_detect as ld

class TestModule(unittest.TestCase):

    def test_file_import(self):
        file = ld.df
        self.assertIsNotNone(file)
    
    def test_pickle_imports(self):
        model = ld.naive_model
        data = ld.data
        self.assertTrue(data)
        self.assertTrue(model)

    def test_prediction(self):
        prediction = ld.lang_prediction('Hello World')
        self.assertEqual(prediction, 'English')
        prediction = ld.lang_prediction('Como estas')
        self.assertEqual(prediction, 'Spanish')
         
if __name__ == '__main__':
    unittest.main()