from .models import Reader
from django.forms import ModelForm

class ReaderForm(ModelForm):
	class Meta:
		model = Reader
		fields = '__all__'