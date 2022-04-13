const express = require('express')
const author = require('../models/author')
const Book = require('../models/books')
const Author = require('../models/author')
const router = express.Router()
router.get('/',async (req,res)=>{
    let searchOptions={}
    if(req.query.name != null && req.query.name !== ''){
        searchOptions.name = new RegExp(req.query.name,'i')
    }
    try{
      const authors = await  Author.find(searchOptions)
      if(!authors) res.redirect('/')
      res.render('authors/index',{
          authors:authors,
        searchOptions:req.query})
    }
    catch(err){
        res.redirect('back')
    }
    
})
router.get('/new',(req,res)=>{
    res.render('authors/new',{author:new Author()})
})
router.post('/',async(req,res)=>{
   try{
   const author= await Author.create({
       name:req.body.name
   })
   
   
   res.redirect(`authors/${author.id}`)
}
catch(err){
console.log('Error in creating author')
res.render('authors/new',{author: author})
}
})
router.get('/:id', async (req,res)=>{
    try{
        const author =  await Author.findById(req.params.id)
     const books = await Book.find({author: author.id}).limit(6).exec()
           res.render('authors/show',{author: author, books:books})
    
   
    }
    catch(e){
        console.log(e)
            res.redirect('back')
    }
})
router.get('/:id/edit',async (req,res)=>{
try{
     const author =  await Author.findById(req.params.id)

    res.render('authors/edit',{author: author})
}
 catch(err){
     console.error(err)
     res.redirect('back')
 }
    
})
router.put('/:id',async (req,res)=>{
    let author 
    try{
       
        author = await Author.findById(req.params.id)
        author.name = req.body.name
        await author.save()


        res.redirect(`/authors/${author.id}`)
     }
     catch {
         if(author== null){ res.redirect('/')
     console.log('Error in creating author')
         }
     else{
        res.render('authors/edit',{author:author})
     }
     }
})
router.delete('/:id', async (req, res) => {
    let author
    try {
      author = await Author.findById(req.params.id)
      await author.remove()
      res.redirect('/authors')
    } catch {
      if (author == null) {
        res.redirect('/')
      } else {
        res.redirect(`/authors/${author.id}`)
      }
    }
  })
module.exports= router