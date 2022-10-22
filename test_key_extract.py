import unittest
import key_extract_hashtag as ke

class TestModule(unittest.TestCase):

    def test_clean(self):
        clean = ke.clean('Sh@nke5H Ra!u M5')
        acutual_result = 'shnkeh rau m'
        self.assertEqual(clean, acutual_result)

    def test_extract(self):
        keywords = ke.extract('shnkeh rau m')
        self.assertIsNotNone(keywords)
      
if __name__ == '__main__':
    unittest.main()