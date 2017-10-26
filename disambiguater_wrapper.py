import json
import disambiguater

class DisambiguaterWrapper():
    def disambiguate(self, input):
        processor =  disambiguater.DemoMRFDisambiguater()
        processor.initMRFDisambiguater()
        return processor.disambiguate(input)
