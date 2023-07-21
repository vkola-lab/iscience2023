# Author: Akshara Balachandra
# Date: Thursday 26 November 2020
# Description: Module to read and parse XML output of Cat12

import xmltodict
from typing import Dict, List


class Cat12Reader:

    def __init__(self, filename: str = None):
        self._filename = filename

    def parseXML(self, key1: str, key2: str) -> zip:
        xmlDict = xmltodict.parse(self._readFile())

        # extract atlas labels
        labels = xmlDict['S'][key1]['names']['item']

        # extract values
        dataString = xmlDict['S'][key1]['data'][key2]
        data = self._parseDataString(dataString)

        return zip(labels, data)

    def parseImageStats(self, key1: str) -> float:
        xmlDict = xmltodict.parse(self._readFile())

        return float(xmlDict['S']['subjectmeasures'][key1])

    def extractROINames(self, atlas: str) -> Dict[int, str]:
        xmlDict = xmltodict.parse(self._readFile())
        labels = xmlDict['S'][atlas]['names']['item']
        ids = [int(i) for i in
               self._parseDataString(xmlDict['S'][atlas]['ids'])]

        return dict(zip(ids, labels))

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, newFile: str) -> None:
        self._filename = newFile

    def _readFile(self) -> str:
        fileContents = ''

        with open(self._filename, 'r') as file:
            fileContents = file.read()

        return fileContents

    def _parseDataString(self, dataString: str) -> List[float]:
        strippedData = dataString[1:len(dataString) - 1]
        strippedData = [float(x) for x in strippedData.split(';')]
        return strippedData
