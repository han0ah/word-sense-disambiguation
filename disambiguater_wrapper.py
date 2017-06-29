import json
import disambiguater

class DisambiguaterWrapper():
    def disambiguate(self, input):
        processor =  disambiguater.BaselineDisambiguater()
        return processor.disambiguate()


