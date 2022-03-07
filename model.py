from transformers import AutoModelForSeq2SeqLM
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator

class T5model(Model):
    def __init__(self, pretrainedfile,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(T5model, self).__init__(None, None)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrainedfile)
        self.config = self.model.config
        InitializerApplicator(self)
    
    def init(self):
        self.model = AutoModelForSeq2SeqLM.from_config(self.config)

    def forward(self,inputid,mask=None,outputid=None,outmask=None,label=None,**kargs):
        inputid = inputid.long()
        if label is not None:
            outputid = outputid.long()
            label = label.masked_fill(label==-1,-100)
            output_dict = self.model(input_ids = inputid,attention_mask=mask, decoder_input_ids=outputid, decoder_attention_mask=outmask, labels=label, return_dict=True)
        else:
            return {'output':self.model.generate(inputid,max_length=200)}
        return output_dict