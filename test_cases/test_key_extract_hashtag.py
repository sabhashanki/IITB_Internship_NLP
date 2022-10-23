import unittest
import sys
sys.path.append('Modules')
import key_extract_hashtag as ke

sample_data = """

Oxygen is the chemical element with the symbol O and atomic number 8. 
It is a member of the chalcogen group in the periodic table, a highly 
reactive nonmetal, and an oxidizing agent that readily forms oxides with 
most elements as well as with other compounds. Oxygen is Earth's most 
abundant element, and after hydrogen and helium, it is the third-most 
abundant element in the universe. At standard temperature and pressure, 
two atoms of the element bind to form dioxygen, a colorless and odorless 
diatomic gas with the formula O2. Diatomic oxygen gas currently constitutes 
20.95% of the Earth's atmosphere, though this has changed considerably over 
long periods of time. Oxygen makes up almost half of the Earth's crust in 
the form of oxides.

"""
class TestModule(unittest.TestCase):

    def test_clean(self):
        clean = ke.clean('Sh@nke5H Ra!u M5 We!c0me B@ck')
        acutual_result = 'shnkeh rau m wecme bck'
        self.assertEqual(clean, acutual_result)

    def test_extract(self):
        keywords = ke.extract(sample_data)
        self.assertIsNotNone(keywords)
        self.assertEqual(len(keywords), 5)
    
    def test_hashtag(self):
        hashtags = ke.hashtagg(sample_data)
        self.assertIsNotNone(hashtags)
        self.assertEqual(len(hashtags), 5)
    
if __name__ == '__main__':
    unittest.main()