"""
Process and combine sentiment datasets for training.
"""
import logging
from pathlib import Path
import pandas as pd
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class SentimentProcessor:
    """Process and combine sentiment datasets for training."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Base directory for data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_sec_sentiment(self) -> pd.DataFrame:
        """Process SEC filings sentiment dataset."""
        logger.info("Processing SEC sentiment dataset")
        
        # Load the data (using parquet for efficiency)
        df = pd.read_parquet(self.raw_dir / "stocknewseventssentiment-snes-10" / "data.parquet")
        
        # Create sentiment scores from positive and negative sentiment columns
        df["label"] = df.apply(
            lambda row: 1 if row["News - Positive Sentiment"] > row["News - Negative Sentiment"]
            else -1 if row["News - Negative Sentiment"] > row["News - Positive Sentiment"]
            else 0,
            axis=1
        )
        
        # Create content from available news indicators
        news_cols = [
            "News - New Products", "News - Layoffs", "News - Analyst Comments",
            "News - Stocks", "News - Dividends", "News - Corporate Earnings",
            "News - Mergers & Acquisitions", "News - Store Openings",
            "News - Product Recalls", "News - Adverse Events",
            "News - Personnel Changes", "News - Stock Rumors"
        ]
        
        # Create a description of the news events
        df["content"] = df.apply(
            lambda row: ". ".join([
                f"{col.replace('News - ', '')} event detected"
                for col in news_cols
                if row[col] > 0
            ]),
            axis=1
        )
        
        # Keep only rows with actual content
        df = df[df["content"].str.len() > 0][["content", "label"]].copy()
        
        logger.info(f"Processed {len(df)} SEC sentiment records")
        return df
        
    def process_self_labeled(self) -> pd.DataFrame:
        """Process self-labeled sentiment dataset."""
        logger.info("Processing self-labeled sentiment dataset")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        self.raw_dir / "sentiment-classification-selflabel-dataset" / "self_label.csv",
                        encoding=encoding
                    )
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.warning("Could not read self-labeled dataset with any encoding")
                return pd.DataFrame(columns=['content', 'label'])
            
            # Keep relevant columns and rename
            df = df[["text", "sentiment"]].copy()
            df = df.rename(columns={"text": "content", "sentiment": "label"})
            
            # Map sentiment labels to consistent format
            sentiment_map = {
                "positive": 1,
                "negative": -1,
                "neutral": 0
            }
            df["label"] = df["label"].map(sentiment_map)
            
            logger.info(f"Processed {len(df)} self-labeled records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing self-labeled dataset: {str(e)}")
            return pd.DataFrame(columns=['content', 'label'])
        
    def process_stock_tweets(self) -> pd.DataFrame:
        """Process stock market tweets sentiment dataset."""
        logger.info("Processing stock tweets sentiment dataset")
        
        try:
            df = pd.read_csv(self.raw_dir / "stock-market-tweets-labelled-with-gcp-nlp" / "Labelled Tweets.csv")
            
            # Keep relevant columns and rename
            df = df[["full_text", "score"]].copy()
            df = df.rename(columns={"full_text": "content", "score": "label"})
            
            # Map sentiment scores to consistent format (-1, 0, 1)
            df["label"] = df["label"].apply(
                lambda x: 1 if x > 0 else -1 if x < 0 else 0
            )
            
            logger.info(f"Processed {len(df)} stock tweet records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing stock tweets dataset: {str(e)}")
            return pd.DataFrame(columns=['content', 'label'])
        
    def process_financial_news(self) -> pd.DataFrame:
        """Process aspect-based financial news sentiment dataset."""
        logger.info("Processing financial news sentiment dataset")
        
        try:
            df = pd.read_csv(self.raw_dir / "aspect-based-sentiment-analysis-for-financial-news" / "SEntFiN-v1.1.csv")
            
            # Keep relevant columns and rename
            df = df[['Words', 'Decisions']].copy()
            df = df.rename(columns={'Words': 'content', 'Decisions': 'label'})
            
            # Map sentiment labels
            sentiment_map = {
                'Positive': 1,
                'Negative': -1,
                'Neutral': 0
            }
            df['label'] = df['label'].map(sentiment_map)
            
            logger.info(f"Processed {len(df)} financial news records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing financial news dataset: {str(e)}")
            return pd.DataFrame(columns=['content', 'label'])
    
    def process_twitter_financial(self) -> pd.DataFrame:
        """Process Twitter financial news sentiment dataset."""
        logger.info("Processing Twitter financial sentiment dataset")
        
        try:
            # Load both training and validation datasets
            train_df = pd.read_csv(self.raw_dir / "twitter-financial-news-sentiment-dataset" / "sent_train.csv")
            valid_df = pd.read_csv(self.raw_dir / "twitter-financial-news-sentiment-dataset" / "sent_valid.csv")
            
            # Combine datasets
            df = pd.concat([train_df, valid_df], ignore_index=True)
            
            # Keep relevant columns and rename
            df = df[['text', 'label']].copy()
            df = df.rename(columns={'text': 'content'})
            
            # Map sentiment labels (already in correct format)
            df['label'] = df['label'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            
            logger.info(f"Processed {len(df)} Twitter financial records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing Twitter financial dataset: {str(e)}")
            return pd.DataFrame(columns=['content', 'label'])
    
    def process_tweet_impact(self) -> pd.DataFrame:
        """Process tweet sentiment impact dataset."""
        logger.info("Processing tweet sentiment impact dataset")
        
        try:
            df = pd.read_csv(self.raw_dir / "tweet-sentiment-s-impact-on-stock-returns" / "tweets.csv")
            
            # Keep relevant columns and rename
            df = df[['text', 'label']].copy()
            df = df.rename(columns={'text': 'content'})
            
            # Map sentiment labels (already in correct format)
            df['label'] = df['label'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            
            logger.info(f"Processed {len(df)} tweet impact records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing tweet impact dataset: {str(e)}")
            return pd.DataFrame(columns=['content', 'label'])
    
    def combine_datasets(self) -> pd.DataFrame:
        """Combine all processed datasets."""
        logger.info("Combining datasets")
        
        # Process each dataset
        dfs = [
            self.process_sec_sentiment(),
            self.process_self_labeled(),
            self.process_stock_tweets(),
            self.process_financial_news(),
            self.process_twitter_financial(),
            self.process_tweet_impact()
        ]
        
        # Combine all datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove any rows with NaN values
        combined_df = combined_df.dropna()
        
        # Save combined dataset
        output_path = self.processed_dir / "combined_sentiment.parquet"
        combined_df.to_parquet(output_path)
        
        logger.info(f"Combined dataset saved to {output_path}")
        logger.info(f"Total records: {len(combined_df)}")
        
        # Print class distribution
        class_dist = combined_df["label"].value_counts()
        logger.info("Class distribution:")
        for label, count in class_dist.items():
            logger.info(f"  {label}: {count} ({count/len(combined_df)*100:.1f}%)")
            
        return combined_df

def main():
    """Process all sentiment datasets."""
    processor = SentimentProcessor()
    processor.combine_datasets()

if __name__ == "__main__":
    main()
