# Generated by Django 4.2.13 on 2024-07-09 11:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedimage',
            name='predicted_class',
            field=models.CharField(blank=True, max_length=10, null=True),
        ),
    ]