from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

print(f"Connecting to database")

# Creating engine
if "postgresql" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,                    
        max_overflow=10,                
        pool_pre_ping=True,             
        pool_recycle=3600,              
        echo=False                     
    )

else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )

# Creating session local class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# Database Models
class LeafPrediction(Base):
    """
    Database model for storing crop leaf disease predictions.
    
    Attributes:
        leaf_id: Primary key, auto-incremented
        crop_type: Type of crop (e.g., Tomato, Potato)
        disease_status: Health status (Healthy/Early_Disease/Severe_Disease)
        confidence_score: Model confidence score (0-1)
        timestamp: time of prediction
        image_filename: filename uploaded
    """
    __tablename__ = "leaf_predictions"
    
    # Primary key
    leaf_id = Column(
        Integer, 
        primary_key=True, 
        index=True, 
        autoincrement=True
    )
    
    # Crop information
    crop_type = Column(
        String(50), 
        nullable=False, 
        index=True,
        comment="Type of crop plant"
    )
    
    # Disease classification
    disease_status = Column(
        String(50), 
        nullable=False,
        index=True,
        comment="Disease status: Healthy, Early_Disease, or Severe_Disease"
    )
    
    # Confidence score
    confidence_score = Column(
        Float, 
        nullable=False,
        comment="Model confidence score (0.0 to 1.0)"
    )
    
    # Timestamp
    timestamp = Column(
        DateTime, 
        default=datetime.utcnow,
        index=True,
        nullable=False,
        comment="Prediction timestamp"
    )
    
    # Image metadata
    image_filename = Column(
        String(255), 
        nullable=True,
        comment="Original uploaded filename"
    )
    
    
    def __repr__(self):
        return f"<LeafPrediction(id={self.leaf_id}, crop={self.crop_type}, status={self.disease_status}, confidence={self.confidence_score:.2f})>"
    
    def to_dict(self):
        """Converting model to dictionary"""
        return {
            "leaf_id": self.leaf_id,
            "crop_type": self.crop_type,
            "disease_status": self.disease_status,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "image_filename": self.image_filename
        }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database initialization
def init_db():
    """
    Initialize database tables.
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
        
        # Print table information
        tables = Base.metadata.tables.keys()
        print(f"Created tables: {', '.join(tables)}")
        
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise


# Test database connection
def test_connection():
    try:
        # Try to connect
        connection = engine.connect()
        connection.close()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
