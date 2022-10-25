import unittest
import module_lang_detect as ld
import logging
logging.basicConfig(level = logging.INFO, filename = 'logs/test_lang_detect.log', filemode = 'w', format = '%(asctime)s - %(levelname)s - %(message)s')
logging.info('All libraries exported')


class TestModule(unittest.TestCase):

    def test_file_import(self):
        file = ld.y
        self.assertIsNotNone(file)
        logging.info('Testing file import module')
    
    def test_pickle_imports(self):
        model = ld.naive_model
        data = ld.data
        self.assertTrue(data)
        self.assertTrue(model)
        logging.info('Testing pickle imports')

    def test_prediction(self):
        prediction = ld.lang_prediction('Hello World')
        self.assertEqual(prediction, 'English')
        prediction = ld.lang_prediction('Como estas')
        self.assertEqual(prediction, 'Spanish')
        logging.info('Testing prediction module')
         
if __name__ == '__main__':
    unittest.main()