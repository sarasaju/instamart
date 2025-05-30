from app import app, db
#from app.models import User, CustomerReview, NewModel  # Import your models

# Create the tables for the default database (customer_reviews.db)
with app.app_context():
  db.create_all()

# Create the tables for the new database (database.db)
#db.create_all(bind='newdb')
