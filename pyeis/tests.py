from __future__ import absolute_import

import unittest
import sympy
from . import circuit_decomposition as cdp
from . import errors

class CircuitSymbolicTest(unittest.TestCase):

    def setUp(self):
        self.circuit = 'Rel+Cdl/(Rct+Wdl)'

    def tearDown(self):
        self.circuit = 'None'

    def test_isSymbol(self):
        Zs = cdp.get_symbolic_immittance(self.circuit)
        self.assertIsInstance(Zs, sympy.Add)

    def test_raiseUnknownElement(self):
        circuit = 'Sel+Cdl/(Rct+Wdl)'
        self.assertRaises(errors.UnknownComponentError, cdp.get_symbolic_immittance, circuit)

    def test_raiseBracketMismatch(self):
        circuit = 'Rel+Cdl/Rct+Wdl)'
        self.assertRaises(errors.BracketMismacthError, cdp.get_symbolic_immittance, circuit)

if __name__ == '__main__':
    unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)