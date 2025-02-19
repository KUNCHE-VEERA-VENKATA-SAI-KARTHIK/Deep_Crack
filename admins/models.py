from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'

        


from django.db import models

class rfcn_b(models.Model):
    name = models.CharField(max_length=255)
    validation_loss = models.FloatField()
    validation_accuracy = models.FloatField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    classification_report = models.TextField()
    confusion_matrix = models.TextField()  # Stored as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'rfcn_b'

    












